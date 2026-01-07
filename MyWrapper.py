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

import simulation

class ISMPC2gym_env_wrapper(gym.Env):
  '''
  Class that is a wrapper for the ismpc environment for converting it in a gymnasium environment
  
  :var name: The name of the environment as a string
  :vartype name: str

  :var max_step: maximum steps for the simulations then truncate
  :vartype max_step: int

  :var world: World of Dartpy in wich the simulation take place
  :vartype world: dart.simulation.World
  
  :var viewer: Viewer of Dartpy for render the simulation
  :vartype viewer: dart.gui.osg.Viewer

  :var node: The node of Dartpy of the controller and the robot
  :vartype node: Hrp4Controller

  :var render: Set to True if want to render the simulation (Only True for now)
  :vartype render: bool 

  :var render_rate: Render the simulation each \"render_rate\" steps 
  :vartype render_rate: int

  :var show_plot: Set to True if want to update the plots each \"plot_rate\" steps (Very very slow, unfacible)
  :vartype show_plot: bool 

  :param plot_rate: Update plots each \"plot_rate\" steps 
  :type plot_rate: int

  :var verbose: Set to true id want some text outputs 
  :vartype verbose: bool

  :var previous_states: List of all the previous states of the system
  :vartype previous_states: list[dict[str, Any]]

  :var previous_actions: List of all the previous actions gived to the system
  :vartype previous_actions: list[dict[str, float]]

  :var previous_rewards: List of all the previous rewards for the actor
  :vartype previous_rewards: list[float] 

  :var REWARD_FUNC_CONSTANTS: Dictionary containing all the constants needed for computing the reward function
  :vartype REWARD_FUNC_CONSTANTS: dict[str, float] 
  '''

  name        : str
  max_steps   : int
  render_     : bool 
  render_rate : int
  verbose     : bool

  world : dart.simulation.World
  viewer : dart.gui.osg.Viewer
  node : simulation.Hrp4Controller

  previous_states  : list[ dict[str, any  ] ]
  previous_actions : list[ dict[str, float] ]
  previous_rewards : list[ float            ]

  REWARD_FUNC_CONSTANTS = {
          'r_alive' : 5.0,
    
            'w_ZmP' : 0.3,
        'sigma_ZmP' : 0.1,
        'w_ZmP_dot' : 0.3,
    'sigma_ZmP_dot' : 0.1,
    
          'w_gamma' : 1.0,
      'sigma_gamma' : 0.8,
    
             'w_ZH' : 1.0,
            'w_phi' : 1.0,

         'w_smooth' : 1.0,
     'sigma_smooth' : 0.1,

     'CoM_H_perc_safe' : 0.1
  }


  def __init__(self, 
               name        : str  = 'hrp4', 
               max_step    : int  = 10_000,
               render      : bool = True,
               render_rate : int  = 5, 
               show_plot   : bool = False,
               plot_rate   : int  = 100,
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

    :param show_plot: Set to True if want to update the plots (very very slow option)
    :type show_plot: bool 

    :param plot_rate: Update plots each \"plot_rate\" steps 
    :type plot_rate: int

    :param verbose: Set to true id want some text outputs 
    :type verbose: bool

    :param 
    '''
    # init the name state and maximum steps for the simulations then reset the environment
    self.name        = name
    self.max_steps   = max_step
    self.render_     = render
    self.render_rate = render_rate
    self.show_plot   = show_plot
    self.plot_rate   = plot_rate
    self.verbose     = verbose
    
    self.reset()

    # size of the observation and action spaces
    self.obs_size = len(self.GetState()[0]) # automatically take the length of the state
    self.action_size = 3 # the action shuld be the displacement alog x y and angular: [Dx, Dy, Dtheta]
    
    # define the observation and action spaces as box without range
    self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (self.obs_size,)   , dtype = np.float64) 
    self.action_space      = gym.spaces.Box(low = -2e-3     , high =2e-3     , shape = (self.action_size,), dtype = np.float64) # action space must be limited
    
    if self.verbose: print(f'environment \"{self.name}\" initialized')

  def step(self, action : np.array) -> tuple[np.array, float, bool, bool, dict[str, any]]:
    '''
    Method for taking a step in the environment performing the \"action\" and computing the reward.
    
    :param action: The action that the agent want to take in the environemnt, in our case the deviation of the footsteps
    :ytpe action: np.array
    :return: state: the new state reached in the environment taking the action |
             reward: The reward earned for taking the action in the previous state and reaching the current state |
             terminated: Condition if simulation is terminated for unhelty condition
             truncated: Condition if simulation is truncated because too long |
             info: Dictionary containinf usefool informations
    :rtype: tuple [state: np.array | reward: float | termination: bool | truncation: bool | info: dict[str, Any]]
    '''

    if self.verbose: print(f"taking a step using action: {action}")

    # convert and use the action
    action_dict = self.PreprocessAction(action)

    # get termination or truncation conditions
    terminated = False                                # troncate because of unhelty conditions
    try:
      # take a step in to the environment
      if self.node.footstep_planner.get_step_index_at_time(self.node.time) >= 1: # start to modify after the 6 step of the robot
        self.ApplyAction(action_dict)

      self.node.customPreStep()
      self.world.step()
    except Exception as e:
      print("Failure during simulation")
      print(e)
      terminated = True

    # collect the state and the reward
    state_array, state_dict = self.GetState()
    reward = self.GetReward(state_dict, action_dict)
    print("Reward: "+ str(reward))

    # update the current step counter
    self.current_step += 1

    truncated = self.current_step > self.max_steps   # truncate the termination because to long

    # render and plot updating
    self.render()

    # log and plot
    if self.show_plot:
      self.UpdatePlot()

    info = {'state' : state_dict, 'reward' : reward, 'steps' : self.current_step, 'max_steps' : self.max_steps}
    return state_array, reward, terminated, truncated, info

  def reset(self, *, seed : int | None = None, options = None,) -> tuple[np.array, dict[str, any]]:
    '''
    Method for reset the simulation to the initial paramethers
    
    :param seed: The seed if the initialization is random
    :type seed: int | None
    :param options: Read the documentation of gymnasium env
    :return: The resetted state for begin the simulation and the dictionary with interesting infos
    :rtype: tuple [state: Any | info: dict[str, Any]]
    '''

    self.world, self.viewer, self.node = simulation.simulation_setup(self.render_)
    self.is_plot_init = False

    # reset the lists that store the previous states and actions usefool for computing the rewards
    self.previous_states  = []
    self.previous_actions = [ {'Dx' : 0., 'Dy' : 0., 'Dth': 0.}]
    self.previous_rewards = []

    # reset the states and steps
    state_array, state_dict = self.GetState()
    self.current_step = 0

    info = {'current steps' : self.current_step, 'max steps' : self.max_steps}

    if self.verbose: print("env resetted")

    return state_array, info

  def render(self) -> None:
    '''
    Method that call the render function for the simulation and update the plot
    '''
    if self.verbose: print('Rendering the simulation')

    if self.render_ and self.current_step % self.render_rate == 0: 
      self.node.RenderFootsteps()
      self.viewer.frame()

  def close(self) -> None:
    # nothing should be done for correctly close the environment
    print('Environment closed, to be implemented')

# UTILS METHODS FOR EXTRACTING INFORMATION FROM THE ENVIRONEMNT AND PROCESS DATA

  def GetState(self) -> tuple[np.array, dict[str, any]]:
    '''
    Function for computing the current state of the system both as a np.array for the policy neural network and 
    as dictionary for easy access in fourther use like in the reward computation. 
    The state is automatically inserted in the list of previous states
    
    :return: A tuple of the current state as a np.array ready to be given to the policy neural network
             and a dictionary of all the usefool therms for compute the reward.

    :rtype: tuple[np.array, dict[str, Any]]
    '''

    # shorthands
    step_index = self.node.footstep_planner.get_step_index_at_time(self.node.time)
    plan = self.node.footstep_planner.plan
    # which foot is used for support (\sigma_k)
    support_foot = 1 if plan[step_index]['foot_id'] == 'rfoot' else -1
    pivot = self.node.lsole if support_foot == 'lfoot' else self.node.rsole
    # remaining time in swing (\T_r_k)
    remaining_time = self.node.footstep_planner.get_remaining_time_in_swing(self.node.time)
    # ISMPC state (com, zmp, torso, base, feet, position and velocity)
    ismpc_state = self.node.retrieve_state(pivot)
    com_pos = ismpc_state['com']['pos']
    
    # mass, gravity, angular momentum of the centroidal dynamics around the support pivot
    mass = self.node.hrp4.getMass()
    g = self.node.params['g']

    # # angular momentum about support pivot
    L = self.node.compute_angular_momentum(pivot.getTransform().translation())
    # derivatives of angular momentum
    Ldot_x = -mass*g*com_pos[1]
    Ldot_y = mass*g*com_pos[0]
    Ldot = np.array([Ldot_x, Ldot_y])

    # position of support foot
    support_foot_pos = plan[step_index]['pos']
    # position of next footstep (taken by foot opposing the current support foot)
    next_footstep_pos = plan[step_index + 1]['pos']
    # position of next footstep that will be taken by the current support foot
    # (second footstep from now)
    support_foot_next_pos = plan[step_index + 2]['pos']

    # compute the state as a np.array and as a dictionary
    state_dict = {
      'support_foot': np.array([support_foot]),
      'remaining_time': np.array([remaining_time]),
      'com_pos':  ismpc_state['com']['pos'],
      'com_vel':  ismpc_state['com']['vel'],
      'zmp_pos':  ismpc_state['zmp']['pos'],
      'zmp_vel':  ismpc_state['zmp']['vel'],
      'torso_orient': ismpc_state['torso']['pos'],
      'torso_angvel': ismpc_state['torso']['vel'],
      'base_orient': ismpc_state['base']['pos'],
      'base_angvel': ismpc_state['base']['vel'],
      'zmp_pos_desired': self.node.desired['zmp']['pos'],
      'zmp_vel_desired': self.node.desired['zmp']['vel'],
      'angular_momentum': L,
      'angular_momentum_drv': Ldot,
      'support_foot_pos': support_foot_pos,
      'support_foot_next_pos': support_foot_next_pos,
      'next_footstep_pos': next_footstep_pos,
      'previous_action': list(self.previous_actions[-1].values())
    }
    state_array = np.concatenate(list(state_dict.values()))

    # store the new state dict in the list of previous states
    self.previous_states.append(state_dict)
    return state_array, state_dict
  
  def PreprocessAction(self, action : np.array) -> dict[str, float]:
    '''
    Method for preprocess the action gived from the policy neural network (probably a np.array) 
    to an action that can be given to the environment (probably a list of float). 
    The action is automatically iserted int he list of previous actions
    
    :param action: np.array of action gived from the policy neural network
    :type action: np.array

    :return: The action as dictonary of float that can be used to perturbate the current footsteps
    :rtype: dict[str, float]
    '''

    # compute the current action as a dictionary
    action_dict = {'Dx' : action[0],
                   'Dy' : action[1],
                   'Dth': action[2]}

    # add the current action dict to the list of previous actions
    self.previous_actions.append(action_dict)

    return action_dict
  
  def ApplyAction(self, action_dict : dict[str, float]) -> None:
    '''
    Method for applyng the action to the environemnt. More in detail this method modify the footstep plan
    according to the actions given.
    
    :param action_dict: The dictionary containing the displacements Dx, Dy and Dtheta
    :type action_dict: dict[str, float]
    '''

    # to be removed the modulation of the displacement
    pos_displacement = np.array([action_dict['Dx'], action_dict['Dy'], 0.0])*self.node.footstep_planner.get_normalized_remaining_time_in_swing(self.node.time)
    ang_displacement = np.array([0.0, 0.0, action_dict['Dth']])
    self.node.footstep_planner.modify_plan(pos_displacement, ang_displacement, self.node.time)

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
    current_reward = 0.0
    
    # if not enough state for compute the reward return 0
    if len(self.previous_states) < 2:
      if self.verbose: print("Not enough states to compute the reward, returning 0")
      return current_reward

    # if not in safe set reward = 0 and return
    if not self.Is_in_Safe_Set(state):
      print("Not in safe set for reward")
      self.previous_rewards.append(current_reward)
      return current_reward

    if state["support_foot"][0] == self.previous_states[-2]["support_foot"][0]: 
      current_reward += self.R_sw(state, action)  
    else:
      current_reward += self.R_end(state, action)

    # add the current reward to the list of previous rewards
    self.previous_rewards.append(current_reward)

    return current_reward

  def UpdatePlot(self) -> None:
    '''
    Method to update the plots at the current time (Very very slow) 
    Suggest: not update frequently during the simulation
    '''
    if not self.is_plot_init:
      self.InitPlot()
    self.node.logger.update_plot(self.node.time)
    
  def InitPlot(self) -> None:
    self.node.logger.initialize_plot(frequency=10)
    self.is_plot_init = True
    
    
  def Is_in_Safe_Set(self, state : dict[str, any]) -> bool:
    
    '''
    Method that check if the current state is in the safe set S

    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :return: True if the state is in the safe set, False otherwise
    :rtype: bool
    '''
    safe_CoM = False
    safe_ZmP = False
    
    # CoM constraints
    Com_Lb = (self.node.mpc.h - self.node.mpc.h * self.REWARD_FUNC_CONSTANTS['CoM_H_perc_safe'])
    Com_Up = (self.node.mpc.h + self.node.mpc.h * self.REWARD_FUNC_CONSTANTS['CoM_H_perc_safe'])

    safe_CoM = state['com_pos'][2] >= Com_Lb and state['com_pos'][2] < Com_Up

    # zmp constraints
    safe_ZmP_x = (state['zmp_pos'][0] <= self.node.mpc.sol.value(self.node.mpc.zmp_x_mid_param)[0] + self.node.params['foot_size'] / 2.) and \
                  (state['zmp_pos'][0] >= self.node.mpc.sol.value(self.node.mpc.zmp_x_mid_param)[0] - self.node.params['foot_size'] / 2.) 
    safe_ZmP_y = (state['zmp_pos'][1] <= self.node.mpc.sol.value(self.node.mpc.zmp_y_mid_param)[0] + self.node.params['foot_size'] / 2.) and \
               (state['zmp_pos'][1] >= self.node.mpc.sol.value(self.node.mpc.zmp_y_mid_param)[0] - self.node.params['foot_size'] / 2.)
    safe_ZmP_z = (state['zmp_pos'][2] <= self.node.mpc.sol.value(self.node.mpc.zmp_z_mid_param)[0] + self.node.params['foot_size'] / 2.) and \
               (state['zmp_pos'][2] >= self.node.mpc.sol.value(self.node.mpc.zmp_z_mid_param)[0] - self.node.params['foot_size'] / 2.)
    safe_ZmP = safe_ZmP_x and safe_ZmP_y and safe_ZmP_z
               
    safe = safe_CoM and safe_ZmP
    if not safe:
      print(f"com safe: {safe_CoM}, zmp safe: {safe_ZmP_x},{safe_ZmP_y},{safe_ZmP_z}")
    return 
  
  
  def R_sw(self, state : dict[str, any],action : dict[str, float]) -> float:
    '''
    Reward function for the swing phase

    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :return: The reward for the swing phase
    :rtype: float
    '''
    
    r_ZmP_x = Ker(state['zmp_pos'][0] - state['zmp_pos_desired'][0], self.REWARD_FUNC_CONSTANTS['sigma_ZmP'], self.REWARD_FUNC_CONSTANTS['w_ZmP'])
    r_ZmP_y = Ker(state['zmp_pos'][1] - state['zmp_pos_desired'][1], self.REWARD_FUNC_CONSTANTS['sigma_ZmP'], self.REWARD_FUNC_CONSTANTS['w_ZmP'])
    r_ZmP_z = Ker(state['zmp_pos'][2] - state['zmp_pos_desired'][2], self.REWARD_FUNC_CONSTANTS['sigma_ZmP'], self.REWARD_FUNC_CONSTANTS['w_ZmP'])

    r_ZmP_dot_x = Ker(state['zmp_vel'][0] - state['zmp_vel_desired'][0], self.REWARD_FUNC_CONSTANTS['sigma_ZmP_dot'], self.REWARD_FUNC_CONSTANTS['w_ZmP_dot'])
    r_ZmP_dot_y = Ker(state['zmp_vel'][1] - state['zmp_vel_desired'][1], self.REWARD_FUNC_CONSTANTS['sigma_ZmP_dot'], self.REWARD_FUNC_CONSTANTS['w_ZmP_dot'])
    r_ZmP_dot_z = Ker(state['zmp_vel'][2] - state['zmp_vel_desired'][2], self.REWARD_FUNC_CONSTANTS['sigma_ZmP_dot'], self.REWARD_FUNC_CONSTANTS['w_ZmP_dot'])
    
    r_gamma = Ker(state['torso_orient'][2] - state['zmp_pos_desired'][0], self.REWARD_FUNC_CONSTANTS['sigma_gamma'],self.REWARD_FUNC_CONSTANTS['w_gamma'])
    
    r_ZH  = - self.REWARD_FUNC_CONSTANTS['w_ZH'] * np.abs(state['com_pos'][2] - self.node.mpc.h)
    r_phi = - self.REWARD_FUNC_CONSTANTS['w_phi'] * np.linalg.norm(state['zmp_pos'][0:1], ord=2)

    r_ZmP = r_ZmP_x + r_ZmP_y + r_ZmP_z
    r_ZmP_dot = r_ZmP_dot_x + r_ZmP_dot_y + r_ZmP_dot_z
    

    return self.REWARD_FUNC_CONSTANTS['r_alive'] + r_ZmP + r_ZmP_dot + r_gamma + r_ZH + r_phi
  
  
  def R_end(self, state : dict[str, any], action : dict[str, float]) -> float:
    '''
    Reward function for the end of the step

    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :return: The reward for the end of the step
    :rtype: float
    '''
    if state["previous_action"] is None:
      if self.verbose: print("Not enough actions to compute the reward, returning 0")
      return 0.0
    
    current_action = np.array([action['Dx'], action['Dy'], action['Dth']])
    previous_action = np.array(state["previous_action"])
    
    e_smooth = current_action - previous_action

    r_smooth1 = Ker(e_smooth[0], self.REWARD_FUNC_CONSTANTS['sigma_smooth'], self.REWARD_FUNC_CONSTANTS['w_smooth'])
    r_smooth2 = Ker(e_smooth[1], self.REWARD_FUNC_CONSTANTS['sigma_smooth'], self.REWARD_FUNC_CONSTANTS['w_smooth'])
    r_smooth3 = Ker(e_smooth[2], self.REWARD_FUNC_CONSTANTS['sigma_smooth'], self.REWARD_FUNC_CONSTANTS['w_smooth'])

    return r_smooth1 + r_smooth2 + r_smooth3