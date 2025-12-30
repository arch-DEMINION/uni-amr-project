import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta
import dartpy as dart
import copy
from utils import *
import os
import ismpc
import footstep_planner
import inverse_dynamics as id
import filter
import foot_trajectory_generator as ftg
from logger import Logger

from simulation import simulation_setup

class ISMPC2gym_env_wrapper(gym.Env):
  '''
  Class that is a wrapper for the ismpc environment for converting it in a gymnasium environment
  
  :var name: The name of the environment as a string
  :vartype name: str

  :var max_step: maximum steps for the simulations then truncate
  :vartype max_step: int

  :var render: Set to True if want to render the simulation (Only True for now)
  :vartype render: bool 

  :var render_rate: Render the simulation each \"render_rate\" steps 
  :vartype render_rate: int

  :var verbose: Set to true id want some text outputs 
  :vartype verbose: bool

  :var previous_states: List of all the previous states of the system
  :vartype previous_states: list[dict[str, Any]]

  :var previous_actions: List of all the previous actions gived to the system
  :vartype previous_actions: list[dict[str, float]]

  :var previous_rewards: List of all the previous rewards for the actor
  :vartype previous_rewards: list[float] 
  '''

  name        : str
  max_steps   : int
  render_     : bool 
  render_rate : int
  verbose     : bool

  previous_states  : list[ dict[str, any  ] ]
  previous_actions : list[ dict[str, float] ]
  previous_rewards : list[ float            ]


  def __init__(self, 
               name        : str  = 'hrp4', 
               max_step    : int  = 10_000,
               render       : bool = True,
               render_rate : int  = 5, 
               verbose     : bool = False):
    '''
    Class that wrap gymnasium environment for taking steps in to a dartpy simulation defined in \"simulation.py\"
    
    :param name: Name of the simulation, default is \"hrp4\"
    :type name: str

    :param max_step: maximum steps for the simulations then truncate
    :type max_step: int

    :param render: Set to True if want to render the simulation (Only True for now)
    :type render: bool 

    :param render_rate: Render the simulation each \"render_rate\" steps 
    :type render_rate: int

    :param verbose: Set to true id want some text outputs 
    :type verbose: bool
    '''
    # init the name state and maximum steps for the simulations then reset the environment
    self.name = name
    self.max_steps = max_step
    self.render_ = render
    self.render_rate = render_rate
    self.verbose = verbose

    self.reset()
    if self.verbose: print(f'environment \"{self.name}\" initialized')

  def step(self, action : torch.tensor) -> tuple[torch.tensor, float, bool, bool, dict[str, any]]:
    '''
    Method for taking a stem in the environment performing the \"action\" and computing the reward.
    
    :param action: The action that the agent want to take in the environemnt, in our case the deviation of the footsteps
    :ytpe action: torch.tensor
    :return: state: the new state reached in the environment taking the action |
             reward: The reward earned for taking the action in the previous state and reaching the current state |
             terminated: Condition if simulation is terminated for unhelty condition
             truncated: Condition if simulation is truncated because too long |
             info: Dictionary containinf usefool informations
    :rtype: tuple [state: torch.tensor | reward: float | termination: bool | truncation: bool | info: dict[str, Any]]
    '''

    if self.verbose: print(f"taking a step using action: {action}")

    # convert and use the action
    action_dict = self.PreprocessAction(action)

    # take a step in to the environment
    self.node.customPreStep()
    self.world.step()

    # collect the state and the reward
    state_tensor, state_dict = self.GetState()
    reward = self.GetReward(state_dict, action_dict)

    # update the current step counter
    self.current_step += 1

    # get termination or truncation conditions
    terminated = False                                # troncate because of unhelty conditions
    truncated = self.current_step > self.max_steps   # truncate the termination because to long

    self.render()
    info = {'state' : state_dict, 'reward' : reward, 'steps' : self.current_step, 'max_steps' : self.max_steps}
    return state_tensor, reward, terminated, truncated, info

  def reset(self, *, seed : int | None = None, options = None,) -> tuple[torch.tensor, dict[str, any]]:
    '''
    Method for reset the simulation to the initial paramethers
    
    :param seed: The seed if the initialization is random
    :type seed: int | None
    :param options: Read the documentation of gymnasium env
    :return: The resetted state for begin the simulation and the dictionary with interesting infos
    :rtype: tuple [state: Any | info: dict[str, Any]]
    '''

    self.world, self.viewer, self.node = simulation_setup(self.render_)

    # reset the lists that store the previous states and actions usefool for computing the rewards
    self.previous_states  = []
    self.previous_actions = []
    self.previous_rewards = []

    # reset the states and steps
    state_tensor, state_dict = self.GetState()
    self.current_step = 0

    info = {'current steps' : self.current_step, 'max steps' : self.max_steps}

    if self.verbose: print("env resetted")

    return state_tensor, info

  def render(self) -> None:
    if self.verbose: print('Rendering the simulation')
    if not self.render_: return
    if self.current_step % self.render_rate == 0: 
        self.viewer.frame()

  def close(self) -> None:
    # notthing shuld be done for correctly close the environment
    print('Environment closed, to be implemented')

# UTILS METHODS FOR EXTRACTING INFORMATION FROM THE ENVIRONEMNT AND PROCESS DATA

  def GetState(self) -> tuple[torch.tensor, dict[str, any]]:
    '''
    Function for computing the current state of the system both as a tensor for the policy neural network and 
    as dictionary for easy access in fourther use like in the reward computation. 
    The state is automatically inserted in the list of previous states
    
    :return: A tuple of hhe current state as a torch.tensor ready to be given to the policy neural network
             and a dictionary of all the usefool therms for compute the reward.

    :rtype: tuple[torch.tensor, dict[str, Any]]
    '''

    # compute the state as a tensor and as a dictionary
    state_tensor = None
    state_dict = None

    # store the new state dict in the list of previous states
    self.previous_states.append(state_dict)
    return state_tensor, state_dict
  
  def PreprocessAction(self, action : torch.tensor) -> dict[str, float]:
    '''
    Method for preprocess the action gived from the policy neural network (probably a tensor) 
    to an action that can be given to the environment (probably a list of float). 
    The action is automatically iserted int he list of previous actions
    
    :param action: Tensor of action gived from the policy neural network
    :type action: torch.tensor

    :return: The action as dictonary of float that can be used to perturbate the current footsteps
    :rtype: dict[str, float]
    '''

    # copute the current action as a dictionary
    action_dict = None

    # add the current action dict to the list of previous actions
    self.previous_actions.append(action_dict)

    return action_dict
  
  def GetReward(self, state : dict[str, any], action : dict[str, float]) -> float:
    '''
    Method for computing the reward based on the state as dictionary, all the environment and the actions
    
    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :param action: The current action as a dictionary that store all the interesting actions indicized using strings
    :type action: dict[str, float]
    :return: Description
    :rtype: float
    '''
    # compute the current reward
    current_reward = None

    # add the current reward to the list of previous rewards
    self.previous_rewards.append(current_reward)

    return current_reward

