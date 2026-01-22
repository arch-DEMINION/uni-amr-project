from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, Schedule

class NoBiasActionBiasACPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
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
        self.action_net = nn.Linear(in_features=self.mlp_extractor.latent_dim_pi, out_features=4, bias=False)