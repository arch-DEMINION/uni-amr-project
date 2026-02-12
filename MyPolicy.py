from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.distributions import BernoulliDistribution, DiagGaussianDistribution
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, Schedule
import torch

class NoBiasActionBiasACPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        trainable_scaling = False, action_decision = False,
        **kwargs
    ):
        self.trainable_scaling = trainable_scaling
        self.action_decision = action_decision
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        # we want to disable the bias on action layer, without doing strange stuff that breaks load()
        # call superclass _build and let it build a action_net which we will discard
        # https://github.com/DLR-RM/stable-baselines3/blob/8fccf7f1c421deff6b54bd595c430604b24724b0/stable_baselines3/common/policies.py#L595
        # action_net is not reference anywhere else after being created in _build()
        super()._build(lr_schedule=lr_schedule)

        # now replace action_net with out own, no bias
        # from https://github.com/DLR-RM/stable-baselines3/blob/8fccf7f1c421deff6b54bd595c430604b24724b0/stable_baselines3/common/policies.py#L595
        # which leads to (self.action_dist.proba_distribution_net) https://github.com/DLR-RM/stable-baselines3/blob/8fccf7f1c421deff6b54bd595c430604b24724b0/stable_baselines3/common/distributions.py#L536C1-L536C66
        action_dim = self.action_space.shape[0]
        log_std = nn.Parameter(torch.zeros(action_dim)) if (not self.action_decision) else nn.Parameter(torch.zeros(action_dim-1))
        self.log_std = log_std
        
        if self.action_decision:
            
            self.action_net = nn.Linear(in_features=self.mlp_extractor.latent_dim_pi, out_features= action_dim-1, bias=False)
            self.decision_net = nn.Linear(self.mlp_extractor.latent_dim_pi,1,bias=False)
            
        else:
            self.action_net = nn.Linear(in_features=self.mlp_extractor.latent_dim_pi, out_features=action_dim, bias=False)


    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Continuous actions

        mean_actions = self.action_net(latent_pi)
        action_dim = self.action_space.shape[0] if not self.action_decision else self.action_space.shape[0] - 1
        cont_dist = DiagGaussianDistribution(action_dim=action_dim)
        cont_dist = cont_dist.proba_distribution(mean_actions, self.log_std)

        actions_con = cont_dist.get_actions(deterministic=deterministic)
        log_prob_con = cont_dist.log_prob(actions_con)

        # Bernoulli decision
        if  self.action_decision:
            
            decision_logits = self.decision_net(latent_pi)
            decision_probs = torch.sigmoid(decision_logits)
            decision_dist = BernoulliDistribution(action_dims=decision_logits.shape[-1])
            decision_dist = decision_dist.proba_distribution(decision_probs)
            decision = decision_dist.get_actions(deterministic=deterministic)
            log_prob_decision = decision_dist.log_prob(decision)

            # Final actions + log prob
        
            actions = torch.cat([actions_con, decision], dim=-1)
            actions = actions.reshape((-1, *self.action_space.shape))

            log_prob = log_prob_decision + decision.detach() * log_prob_con
        else:
            actions = actions_con
            log_prob = log_prob_con

        # Value function
        values = self.value_net(latent_vf)

        return actions, values, log_prob
    
    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Continuous actions
        action_dim = self.action_space.shape[0] if not self.action_decision else self.action_space.shape[0] - 1
        cont_dist = DiagGaussianDistribution(action_dim=action_dim)
        mean_actions = self.action_net(latent_pi)
        cont_dist = cont_dist.proba_distribution(mean_actions, self.log_std)
        actions_con = actions[:, :action_dim]  
        log_prob_con = cont_dist.log_prob(actions_con)
        entropy_con = cont_dist.entropy()

        if  self.action_decision:
            # Bernoulli decision
            decision_logits = self.decision_net(latent_pi)
            decision_probs = torch.sigmoid(decision_logits)
            decision_dist = BernoulliDistribution(action_dims=decision_logits.shape[-1])
            decision_dist = decision_dist.proba_distribution(decision_probs)
            decision_action = actions[:, action_dim:]  
            log_prob_decision = decision_dist.log_prob(decision_action)
            entropy_decision = decision_dist.entropy()

            # Combine
            log_prob = log_prob_decision + decision_action.detach() * log_prob_con
            entropy = entropy_con + entropy_decision
        else:
            log_prob = log_prob_con
            entropy = entropy_con

        values = self.value_net(latent_vf)
        
        return values, log_prob, entropy