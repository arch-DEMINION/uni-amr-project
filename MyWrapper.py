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
import random
from logger import Logger
import utils
import math

import simulation

import colorama
from termcolor import colored

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
          'r_alive' : 3.0,
      
             'w_ZH' : 0.1,
            'w_phi' : 0.1,

        'w_vel_ref':  0,  # originally 1.5
     'sigma_vel_ref': 0.1,
     
         
          'w_L' : 3.5,
          'sigma_L': 0.05,
     
          'w_footstep' : 0.8,  # originally 10.0
      'sigma_footstep' : 0.15,
'sigma_footstep_bonus' : 0.2,
      'distance_bonus' : 0.35,   # originally 0.45

    'terminated_penalty' : -1000.0,
    'sigma_desired_footstep': 0.1, 
    'omega_desired_footstep': 0,  #originally 2.5

    'action_weight_sw'  : 4.0,  # originally 1.0
    'action_weight_ds'  : 4.0,  # originally 1.0
    'action_damping' : 0.001,
    'end_of_plan' : 1000.0,  # originally 100.0
    'footstep_checkpoint' : 5.0  #originally 3.0
  }

  PERTURBATION_PARAMETERS = {
    'gravity_x_range' : np.array([0.06, 0.12]) * 1, # [3,4°, 6,8°] * scale
    'gravity_y_range' : np.array([0.06, 0.12]) * 1,
    'gravity_change_prob' : 0 * 0.01, # 1%
    'ext_force_appl_prob': 0.00333 * 3.0,  # 1%
    'force_range': np.array([50, 150]) * 0.3,   # Newton
    'CoM_offset_range': np.array([0.001, 0.05]) # meters from the CoM of the body
  }

  COLOR_CODE = {
    'forces'   : 'blue',
    'exception' : 'red',
    'reward'   : 'green',
    'checkpoint' : 'magenta'
  }

  REWARD_LOWER_BOUND = -1500
  LEVELING_SYSTEM = {
    'starting_level'   : 20,
    'exp_to_new_level' : 6,
    'exp_gain' : 2,
    'exp_loss' : 1
  }

  def __init__(self, 
               name        : str  = 'hrp4', 
               max_step    : int  = 1_000,
               render      : bool = True,
               render_rate : int  = 5, 
               show_plot   : bool = False,
               plot_rate   : int  = 100,
               verbose     : bool = False,
               mpc_frequency : int = 10,
               agent_frequency : int = 1,
               frequency_change_grav : int = 1,
               footstep_scaler: float = 0.9,
               action_space_bounds: float = 0.02,
               desired_trajectory: int = -1,
               curriculum_learning: bool = False,
               grav_bool : float= 1.0,
               force_bool : float =1.0,
               get_L_reference : bool = False,
               get_ref_node :bool= False):
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

    :param footstep_scaler: Scaler parameter to use in modifying the footstep plan. 0 means only the next footstep is modified, 1 means the whole plan is displaced
    :type footstep_scaler: float
    '''

    # init the name state and maximum steps for the simulations then reset the environment
    self.name        = name
    self.max_steps   = max_step
    self.render_     = render
    self.render_rate = render_rate
    self.show_plot   = show_plot
    self.plot_rate   = plot_rate
    self.verbose     = verbose
    self.mpc_frequency = mpc_frequency
    self.frequency_change_of_grav = frequency_change_grav
    self.agent_frequency = agent_frequency
    self.footstep_scaler = footstep_scaler
    self.desired_trajectory = desired_trajectory
    self.curriculum_learning = curriculum_learning
    self.get_L_reference = get_L_reference
    self.get_ref_node = get_ref_node
    self.grav_bool = grav_bool
    self.force_bool = force_bool
    self.L_des = []
    
    colorama.init()

    self.end_of_plan_counter = 0
    self.level = self.LEVELING_SYSTEM['starting_level']
    self.episodes = 0
    self.init_gravity_ranges = (self.PERTURBATION_PARAMETERS['gravity_x_range'], self.PERTURBATION_PARAMETERS['gravity_y_range'])
    state , _ = self.reset(first_time_flag = True)

    # size of the observation and action spaces
    self.obs_size = len(state) # automatically take the length of the state
    self.action_size = 3 # the action shuld be the displacement alog x y and angular: [Dx, Dy, Dtheta]
    
    # define the observation and action spaces as box without range
    self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape = (self.obs_size,)   , dtype = np.float64) 
    self.action_space      = gym.spaces.Box(low = -action_space_bounds    , high = action_space_bounds   , shape = (self.action_size,), dtype = np.float64) # action space must be limited

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
    # we consider the safe set to be whenever to termination conditions occur
    terminated = False
    try:
      # take a step in to the environment
      starting_step = self.node.footstep_planner.get_step_index_at_time(self.node.time) # remember the starting step
      start_time = self.node.time
      self.ApplyAction(action_dict)

      # simulate robot and environment using dartpy
      for i in range(self.mpc_frequency):        
        self.node.customPreStep()
        self.world.step()
        self.current_MPC_step += 1
        self.solver_status = self.node.mpc.sol.stats()["return_status"]  
        self.render()
      
      # apply the forces 
      if np.random.random() < self.PERTURBATION_PARAMETERS['ext_force_appl_prob'] * self.force_bool:
        random_force, random_point, random_body, random_body_name = self.Get_random_force(self.PERTURBATION_PARAMETERS['force_range'], self.PERTURBATION_PARAMETERS['CoM_offset_range'])
        random_body.addExtForce(random_force, random_point, True)
        #if self.verbose: print("\nAdded force: " + str(random_force) + " at body: " + str(random_body_name)+ "\n")
        print(colored(f"\nApplied force: {random_force} at body:  {random_body_name} \n", self.COLOR_CODE['forces']))
        self.world.step()
        self.render()
        
      
    except Exception as e:
      self.solver_status = str(e).split("'")[-2]
      print(colored(f"Failure during simulation: {self.solver_status}", self.COLOR_CODE['exception']))
      terminated = True

    # collect the state and the reward
    state_array, state_dict = self.GetState()
    
    reward = 0.0

    if not self.get_L_reference:
      reward = self.GetReward(state_dict, action_dict, terminated)
  
    if self.get_L_reference:
      
        self.get_Ldes(self.node.ref_L[-1]) # appending only the last L_des of the 10 MPC cycles computed in step                                                                     # i.e. the one that will be compared with L computed in Get_state()
        self.node.ref_L = []

    # update the current step counter
    self.current_step += 1

    # truncate the episode after a set number of steps or when the plan has been completed
    truncated = self.current_step > self.max_steps or self.end_of_plan_condition()

    if terminated or truncated:
        print(colored(f"Total Reward of the episode: {np.sum(self.previous_rewards):0.3f} | (x, y): ({self.angle_x:0.4f}, {self.angle_y:0.4f})", self.COLOR_CODE['reward']))
    
    # log and plot
    if self.show_plot:
      self.UpdatePlot()

    # sometimes change the gravity a tiny bit (1/10 of the intended perturbation)
    # this simulates the robot being on an inclined plane
    if np.random.random() < self.PERTURBATION_PARAMETERS['gravity_change_prob'] * self.grav_bool:
      self.ChangeGravity(self.PERTURBATION_PARAMETERS['gravity_x_range']*0.1, self.PERTURBATION_PARAMETERS['gravity_y_range']*0.1, additive = True, apply_gravity=False)
      self.world.setGravity(utils.decompose_gravity(self.angle_x, self.angle_y))
      print(colored(f"gravity: (x, y): ({self.angle_x:0.4f}, {self.angle_y:0.4f})", self.COLOR_CODE['forces']))

    info = {'state' : state_dict, 'reward' : reward, 'steps' : self.current_step, 'max_steps' : self.max_steps}
    return state_array, reward, terminated, truncated, info

  def reset(self, *, seed : int | None = None, options = None, first_time_flag = False) -> tuple[np.array, dict[str, any]]:
    '''
    Method for reset the simulation to the initial paramethers
    
    :param seed: The seed if the initialization is random
    :type seed: int | None
    :param options: Read the documentation of gymnasium env
    :return: The resetted state for begin the simulation and the dictionary with interesting infos
    :rtype: tuple [state: Any | info: dict[str, Any]]
    '''

    # the first steps must be unperturbed for make the solver be able to do it
    pre_value = (self.PERTURBATION_PARAMETERS['ext_force_appl_prob'], self.PERTURBATION_PARAMETERS['gravity_change_prob'])
    self.PERTURBATION_PARAMETERS['ext_force_appl_prob'], self.PERTURBATION_PARAMETERS['gravity_change_prob'] = (0, 0)

    # each time it reach the end 5 times increase the difficulty
    if not first_time_flag:
      self.episodes += 1
      #self.Leveling()

    self.world, self.viewer, self.node = simulation.simulation_setup(self.render_, trajectory=self.desired_trajectory, get_reference=self.get_ref_node)

    self.is_plot_init = False

    # reset the lists that store the previous states and actions useful for computing the rewards
    self.previous_states  = []
    self.previous_actions = [ {'Dx' : 0., 'Dy' : 0., 'Dth': 0., 'list' : np.array([0, 0, 0])}]
    self.previous_rewards = []

    self.angle_x = 0.0
    self.angle_y = 0.0

    # reset the states and steps
    state_array, state_dict = self.GetState()
    self.current_step = 0
    self.current_MPC_step = 0
    
    self.footstep_checkpoint_given = False
    self.initial_foot_dist = np.linalg.norm(self.node.initial['lfoot']['pos'][:3] - self.node.initial['rfoot']['pos'][:3], ord=2)

    # advance in the world until the first foot starts moving
    # this is to avoid having the agent work before MPC starts working and the robot cannot move
    # def robot_moving() -> bool:
    #   foot = self.node.footstep_planner.plan[0]['foot_id']
    #   state = self.node.retrieve_state()
    #   initial = self.node.initial[foot]['pos']
    #   foot_pos = state[foot]['pos']
    #   return foot_pos[5] >= initial[5] + 1e-2
    
    # # TODO: check if this has any impact with changing gravity
    # while not robot_moving():
    #   self.node.customPreStep()
    #   self.world.step()
    #   self.render()

    self.PERTURBATION_PARAMETERS['gravity_x_range'] = self.init_gravity_ranges[0]*self.level*0.01
    self.PERTURBATION_PARAMETERS['gravity_y_range'] = self.init_gravity_ranges[1]*self.level*0.01
    
    # eventually change the gravity
    if (self.episodes % self.frequency_change_of_grav) == 0 and self.grav_bool > 0.0:
      self.ChangeGravity(self.PERTURBATION_PARAMETERS['gravity_x_range'], self.PERTURBATION_PARAMETERS['gravity_x_range'], apply_gravity=False)
      self.world.setGravity(utils.decompose_gravity(self.angle_x, self.angle_y))

    # restore the perturbations
    self.PERTURBATION_PARAMETERS['ext_force_appl_prob'], self.PERTURBATION_PARAMETERS['gravity_change_prob'] = pre_value 
    print("\nStarting episode: " + str(self.episodes) + "\n")

    info = {'current steps' : self.current_step, 'max steps' : self.max_steps}

    if self.verbose: print("environment reset")

    return state_array, info

  def render(self) -> None:
    '''
    Method that call the render function for the simulation and update the plot
    '''
    if self.verbose: print('Rendering the simulation')

    if self.render_ and self.current_MPC_step % self.render_rate == 0: 
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
    original_plan = self.node.footstep_planner.original_plan
    # which foot is used for support (\sigma_k)
    support_foot_str = plan[step_index]['foot_id']
    # embedding
    support_foot = np.array([1., 0.]) if support_foot_str == 'rfoot' else np.array([0.,1.])

    pivot = self.node.lsole if support_foot_str == 'lfoot' else self.node.rsole

    # remaining time in swing (\T_r_k)
    remaining_time = self.node.footstep_planner.get_remaining_time_in_swing(self.node.time)
    # ISMPC state (com, zmp, torso, base, feet, position and velocity)
    ismpc_state = self.node.retrieve_state(pivot)
    com_pos = ismpc_state['com']['pos']
    
    # mass, gravity, angular momentum of the centroidal dynamics around the support pivot
    mass = self.node.hrp4.getMass()
    g = self.node.params['g']

    # # angular momentum about support pivot
    L = self.node.compute_angular_momentum(pivot.getTransform(pivot).translation())

    # pose of support foot as [x,y,theta], relative to support foot
    support_foot_pos = ismpc_state[support_foot_str]['pos']
    support_foot_pos = np.array([support_foot_pos[i] for i in [3, 4, 2]])

    support_foot_gpos = self.node.retrieve_state()[support_foot_str]['pos']

    # position of next footstep (taken by foot opposing the current support foot) relative to the current foot position
    next_footstep_pos = plan[step_index + 1]['pos']
    next_footstep_ang = plan[step_index + 1]['ang'][2]
    # move them in the coordinate frame of the pivot + take only [x, y, theta]
    perr = next_footstep_pos - support_foot_gpos[3:]
    oerr = next_footstep_pos - support_foot_gpos[:3]
    perr_pivot = (pivot.getTransform().matrix()@np.concatenate((perr, np.ones(1))))[0:3]
    oerr_pivot = pivot.getTransform().rotation()@oerr[0:3]
    next_footstep_relpos = np.concatenate((perr_pivot, oerr_pivot))
    next_footstep_relpos = np.array([next_footstep_relpos[i] for i in [0,1,5]])  # x, y, gamma 

    perr = original_plan[step_index]['pos'] - support_foot_gpos[3:]
    oerr = original_plan[step_index]['ang'] - support_foot_gpos[:3]
    pdesired_pivot = (pivot.getTransform().matrix()@np.concatenate((perr, np.ones(1))))[0:3]
    odesired_pivot = (pivot.getTransform().rotation()@oerr)[0:3]
    desired_pivot = np.concatenate((pdesired_pivot, odesired_pivot))
    desired_pivot = np.array([desired_pivot[i] for i in [0, 1, 5]])
    
    ref_vel = pivot.getTransform().rotation()@self.node.footstep_planner.vref[step_index]

    # compute the state as a np.array and as a dictionary
    state_dict = {
      'support_foot': support_foot,
      'remaining_time': np.array([remaining_time]),
      'com_pos':  ismpc_state['com']['pos'],
      'com_vel':  ismpc_state['com']['vel'],
      'zmp_pos':  ismpc_state['zmp']['pos'],
      'ref_vel': ref_vel,
      'zmp_pos_desired': self.node.desired['zmp']['pos'],
      'angular_momentum': L,
      'next_footstep_relpos': next_footstep_relpos,
      'desired_footstep_relpos': desired_pivot,
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
    action_dict = {'Dx'   : action[0],
                   'Dy'   : action[1],
                   'Dth'  : action[2],
                   'list' : action}

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
    
    # get the current angular position of the support footstep
    index = self.node.footstep_planner.get_step_index_at_time(self.node.time)
    current_footestep = self.node.footstep_planner.plan[index]
    z_support_footstep = current_footestep['ang'][2]

    # compute the position displacemnt along the support foot reference
    pos_displacement = np.array([action_dict['Dx'] * cos(z_support_footstep) - action_dict['Dy'] * sin(z_support_footstep),\
                                 action_dict['Dx'] * sin(z_support_footstep) + action_dict['Dy'] * cos(z_support_footstep), 0.0])
    ang_displacement = np.array([0.0, 0.0, action_dict['Dth']])

    # scale if is in ss and time is running out
    if self.node.footstep_planner.get_phase_at_time(self.node.time) == 'ss':
      pos_displacement *= self.node.footstep_planner.get_normalized_remaining_time_in_swing(self.node.time)
      ang_displacement *= self.node.footstep_planner.get_normalized_remaining_time_in_swing(self.node.time)
    
    self.node.footstep_planner.modify_plan(pos_displacement, ang_displacement, self.node.time, scaler=self.footstep_scaler)
    
  def GetReward(self, state : dict[str, any], action : dict[str, float], terminated : bool) -> float:
    '''
    Method for computing the reward based on the state as dictionary, all the environment and the actions
    
    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :param action: The current action as a dictionary that store all the interesting actions indicized using strings
    :type action: dict[str, float]
    :param terminated: The termination state of the robot
    :type terminated: bool
    :return: Description
    :rtype: float
    '''

    # termination penalty and alive bonus
    terminated_penalty =  self.REWARD_FUNC_CONSTANTS['terminated_penalty'] * 0.5 if self.solver_status == 'solved inaccurate'          else \
                          self.REWARD_FUNC_CONSTANTS['terminated_penalty'] * 3.0 if self.solver_status == 'problem non convex'         else \
                          self.REWARD_FUNC_CONSTANTS['terminated_penalty'] * 2.0 if self.solver_status == 'feet_collision'         else \
                          self.REWARD_FUNC_CONSTANTS['terminated_penalty']
    
    current_reward = 0.0 + terminated_penalty if terminated else \
                           self.REWARD_FUNC_CONSTANTS['r_alive']
    
    # if not enough state for compute the reward return 0
    if len(self.previous_states) < 2:
      if self.verbose: print("Not enough states to compute the reward, returning 0")
      self.previous_rewards.append(current_reward)
      return current_reward

    # change reward depending on gait phase
    current_reward += self.R_sw(state, action) if state['remaining_time'] > 0 else self.R_end(state, action)

    # # try to keep the feet at a proper distance to avoid self collisions
    # # penalty for placing the foots to close
    # r_next_footstep = -Ker(np.linalg.norm(state['next_footstep_relpos'][0:2], ord= 2), self.REWARD_FUNC_CONSTANTS['sigma_footstep'], self.REWARD_FUNC_CONSTANTS['w_footstep'])
    # # bonus for separate foot 5*e^((|x| - 0.45)/0.2)^2   
    # r_next_footstep_bonus = Ker(np.abs(np.linalg.norm(state['next_footstep_relpos'][0:2], ord= 2)) - self.REWARD_FUNC_CONSTANTS['distance_bonus'], 
    #                              self.REWARD_FUNC_CONSTANTS['sigma_footstep_bonus'], 
    #                              self.REWARD_FUNC_CONSTANTS['w_footstep']) 
    # current_reward += r_next_footstep + r_next_footstep_bonus
    
    # reward for angular momentum tracking
    if len(self.L_des) != 0:
      
      r_angular_momentum_x = Ker(self.L_des[self.current_step][0]- state['angular_momentum'][0], self.REWARD_FUNC_CONSTANTS['sigma_L'], self.REWARD_FUNC_CONSTANTS['w_L'])
      r_angular_momentum_y = Ker(self.L_des[self.current_step][1]- state['angular_momentum'][1], self.REWARD_FUNC_CONSTANTS['sigma_L'], self.REWARD_FUNC_CONSTANTS['w_L'])
      current_reward += r_angular_momentum_x + r_angular_momentum_y
    
    # checkpoint bonus
    step = self.node.footstep_planner.get_step_index_at_time(self.node.time)
    # reward for end of plan
    if self.end_of_plan_condition():
      current_reward += self.REWARD_FUNC_CONSTANTS['end_of_plan']
      print(colored(f"end of plan reached ({self.LEVELING_SYSTEM['exp_to_new_level'] if   (self.end_of_plan_counter+self.LEVELING_SYSTEM['exp_gain'])%self.LEVELING_SYSTEM['exp_to_new_level'] == 0 else (self.end_of_plan_counter+self.LEVELING_SYSTEM['exp_gain'])%self.LEVELING_SYSTEM['exp_to_new_level']} / {self.LEVELING_SYSTEM['exp_to_new_level']}) | level: {self.level}", 'yellow'))
      self.Leveling()

    # reward for checkpoints in the plan
    # hardcoded every 3rd footstep, except the very first
    if step > 0 or self.end_of_plan_condition():
      if step % 3 == 0 and not self.footstep_checkpoint_given:
        self.footstep_checkpoint_given = True
        current_reward += self.REWARD_FUNC_CONSTANTS['footstep_checkpoint'] * (step * 0.333 * 0.2)
        print(colored(f"reward for reaching step {step}", self.COLOR_CODE["checkpoint"]))
      elif step % 3 > 0:
        self.footstep_checkpoint_given = False

    # reward for CoM following desired velocities (in the same reference frame as the state)
    current_reward += Ker(state['com_vel'][0] - state['ref_vel'][0],  self.REWARD_FUNC_CONSTANTS['sigma_vel_ref'], self.REWARD_FUNC_CONSTANTS['w_vel_ref'])
    current_reward += Ker(state['com_vel'][1] - state['ref_vel'][1],  self.REWARD_FUNC_CONSTANTS['sigma_vel_ref'], self.REWARD_FUNC_CONSTANTS['w_vel_ref'])

    # clip max. negative reward for when ID solver crashes
    current_reward = max(self.REWARD_LOWER_BOUND, current_reward)

    if terminated: self.Leveling()

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
  
  def R_sw(self, state : dict[str, any],action : dict[str, float]) -> float:
    '''
    Reward function for the swing phase

    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :return: The reward for the swing phase
    :rtype: float
    '''
    
    # com_pos and self.node.mpc.h are in the same coordinate systems
    com_pos = self.node.retrieve_state()['com']['pos']
    r_ZH  = - self.REWARD_FUNC_CONSTANTS['w_ZH'] * np.abs(com_pos[2] - self.node.mpc.h)
    # TODO: add rewards/penalties on torso

    # penalty for large displacements near the end of the swing phase
    action_penalty = -self.REWARD_FUNC_CONSTANTS['action_weight_sw']*np.dot(action['list'][::2], action['list'][::2]) / \
                     (self.node.footstep_planner.get_normalized_remaining_time_in_swing(self.node.time) + \
                      self.REWARD_FUNC_CONSTANTS['action_damping'])
                      
    # bonus for placing the footsteps close to the desired position in the original (unmodified) plan
    footstep_bonus = +Ker(np.dot(state['desired_footstep_relpos'], state['desired_footstep_relpos']), self.REWARD_FUNC_CONSTANTS['sigma_desired_footstep'], self.REWARD_FUNC_CONSTANTS['omega_desired_footstep']) 

    return r_ZH + action_penalty + footstep_bonus
  
  def R_end(self, state : dict[str, any], action : dict[str, float]) -> float:
    '''
    Reward function for the end of the step

    :param state: The current state as a dictionary containing all the interesting therms indicized using strings
    :type state: dict[str, any]
    :return: The reward for the end of the step
    :rtype: float
    '''

    # penalize large actions when in double support
    action_penalty = -self.REWARD_FUNC_CONSTANTS['action_weight_ds']*np.dot(action['list'], action['list'])

    return action_penalty 
  
  def ChangeGravity(self, range_x : list[float, float], range_y : list[float, float], additive : bool = False, apply_gravity = True) -> None:
    '''
    Method for changing the gravity in a given interval positive or negative

    :param range_x: The range of x angle [x_min, x_max]
    :type range_x: list[float, float]
    :param range_y: The range of y angle [y_min, y_max]
    :type range_y: list[float, float]
    :param additive: Flag to indicate if is an additive perturbation or a new reset of angles
    :type additive: bool
    '''

    self.angle_y = self.angle_y*additive + np.random.choice([-1, 1]) * ((np.random.random() * (range_y[1] - range_y[0])) + range_y[0])  #0.0
    self.angle_x = self.angle_x*additive + np.random.choice([-1, 1]) * ((np.random.random() * (range_x[1] - range_x[0])) + range_x[0])  # from 3,4° to 6,8°
      
    if apply_gravity: self.world, self.viewer, self.node = simulation.simulation_setup(self.render_, self.angle_x, self.angle_y)
      
  def Get_random_force(self, range_f : list[float, float], range_p : list[float, float]) -> None:
    '''
    Method for applying a random force on a random point of the robot at a certain time step

    :param range_f: The range of admissible modules of forces 
    :type range_f: list[float, float]
    :param range_p: The range of admissible distances of forces application points from the body CoM
    :type range_f: list[float, float]
    '''

    random_force_x = np.random.choice([-1, 1]) * ((np.random.random() * (range_f[1] - range_f[0])) + range_f[0]) 
    random_force_y = np.random.choice([-1, 1]) * ((np.random.random() * (range_f[1] - range_f[0])) + range_f[0])
    random_force_z = np.random.choice([-1, 1]) * ((np.random.random() * (range_f[1] - range_f[0])) + range_f[0])
    
    random_force = np.array([random_force_x , random_force_y, random_force_z])
    
    random_point_x = np.random.choice([-1, 1]) * ((np.random.random() * (range_p[1] - range_p[0])) + range_p[0]) 
    random_point_y = np.random.choice([-1, 1]) * ((np.random.random() * (range_p[1] - range_p[0])) + range_p[0])
    random_point_z = np.random.choice([-1, 1]) * ((np.random.random() * (range_p[1] - range_p[0])) + range_p[0]) 
    
    random_point = np.array([random_point_x , random_point_y, random_point_z])
    
    nodes = {
        "sole": self.node.rsole if self.node.footstep_planner.get_current_footstep_from_plan(self.node.time)['foot_id'] == 0.1 else self.node.lsole, # in the sole that is swinging
        "torso": self.node.torso,
        "body":  self.node.base
    }

    if self.node.footstep_planner.get_phase_at_time(self.node.time) == "ds": nodes.pop('sole') # remove sole if is in double support
    random_body_name, random_body = random.choice(list(nodes.items()))
      
    return random_force, random_point, random_body, random_body_name
  
  def end_of_plan_condition(self):
    return self.node.footstep_planner.get_step_index_at_time(self.node.time) >= (len(self.node.footstep_planner.plan) - 3)
  
  def Leveling(self) -> None:
    if not self.curriculum_learning:
      return

    if self.end_of_plan_condition(): 
        self.end_of_plan_counter += self.LEVELING_SYSTEM['exp_gain']
        if self.end_of_plan_counter >= self.LEVELING_SYSTEM['exp_to_new_level']:
          self.level += 1
          self.end_of_plan_counter = 0
          
          print(colored(f'NEW LEVEL: {self.level}', 'yellow')) 

    else: self.end_of_plan_counter = max(0, self.end_of_plan_counter-self.LEVELING_SYSTEM['exp_loss'])
    
  def set_disturbances(self, grav_bool: float, force_bool: float) -> None:
    '''
    Utility function for setting the disturbances flags

    :param grav_bool: Flag for enabling gravity disturbances
    :type grav_bool: float
    :param force_bool: Flag for enabling external forces disturbances
    :type force_bool: float
    '''
    self.grav_bool = grav_bool
    self.force_bool = force_bool
    return 

  def get_Ldes(self, L_des) -> None:
    '''
    Method for getting the desired angular momentum around the pivot
    :param L_des: The desired angular momentum to append to the reference's list
    :type L_des: list[float, float, float]
    '''
    #self.L_des.extend(L_des)
    self.L_des.append(L_des)
    return
  
  def set_Ldes(self, L_des) -> None:
    '''
    Method for setting the possibility of catching the desired angular momentum around the pivot
    :param L_des: yes or no based on if you want to get it or not
    :type L_des: bool
    '''
    self.get_L_reference = L_des
    return
    
  def compute_Ldes(self) -> None:
    '''
    Method for printing dimentions of the desired angular momentum'slist 
    '''

    print(len(self.L_des), self.L_des[0].shape)
    return
  
