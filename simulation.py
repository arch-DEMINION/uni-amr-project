import numpy as np
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
import timeit
from math import sin,cos,sqrt
import random
import utils as utils
from residual import residual_dynamics


class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4, trajectory=0, get_ref = False):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.get_ref = get_ref
        self.params = {
            'g': 9.81,
            'h': 0.72,
            'foot_size': 0.1,
            'step_height': 0.02,
            'ss_duration': 40,
            'ds_duration': 30,
            'world_time_step': world.getTimeStep(),
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 150, # TO MODIFY originally 250
            'dof': self.hrp4.getNumDofs(),
        }
        self.params['eta'] = np.sqrt(self.params['g'] / self.params['h'])
        self.last_control = -1000
        self.ref_L = []

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

        detector = self.world.getConstraintSolver().getCollisionDetector()
        self.feet_coll_group = detector.createCollisionGroup()
        self.feet_coll_group.addShapeFramesOf(self.rsole)
        self.feet_coll_group.addShapeFramesOf(self.lsole)
        self.feet_coll_option = dart.collision.CollisionOption()
        self.feet_coll_result = dart.collision.CollisionResult()

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

        # initialize footstep planner
        # if trajectory is < 0 pick a random one, except validation ones
        if trajectory <= 0:
            trajectory = random.randint(1, 5)

        if trajectory == 100:
            trajectory = random.randint(101, 103)
            
        if self.get_ref:
            trajectory = 103  # to get always the same reference when getting L_des

        match trajectory:
            case 1:
                # on the spot
                reference = [(0., 0., 0.)] * 10
            case 2:
                # forwards, then backwards
                reference = [(0.10, 0., 0.)] * 15 # + [(-0.1, 0., 0.)] * 10
            case 3:
                # backwards, then forwards
                reference = [(-0.10, 0., 0.)] * 15 # + [(0.1, 0., 0.)] * 10
            # case 3:
            #     # turn on the spot
            #     reference = [(0.0, 0., 0.2)] * 40
            #     self.params['first_swing'] = 'lfoot'
            # case 4:
            #     # turn on the spot (other direction)
            #     reference = [(0.0, 0., -0.2)] * 40
            case 4:
                # to the left, then to the right
                reference = [(0.0, -0.1, 0.)] * 15 + [(0.0, 0.1, 0.)] * 15
            case 5:
                # to the right, then to the left
                reference = [(0.0, 0.1, 0.)] * 15 + [(0.0, -0.1, 0.)] * 15
                # self.params['first_swing'] = 'lfoot'
            # use the following for validation, hence they should never be used in training
            case 101:
                # forward, then to the right
                reference = [(0.15, 0., 0.)] * 15 + [(0.0, 0.1, 0.)] * 15
            case 102:
                # diagonal
                reference = [(0.12, 0.05, 0.)] * 20
                # self.params['first_swing'] = 'lfoot'
            case 103:
                # weird sine like
                reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10  + [(0., 0., 0.)] * 10 
                # self.params['first_swing'] = 'lfoot'


        reference += [(0., 0., 0.)] * 5

        self.plan_skeleton = []
        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )

        # initialize MPC controller
        self.mpc = ismpc.Ismpc(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize kalman filter
        A = np.identity(3) + self.params['world_time_step'] * self.mpc.A_lip
        B = self.params['world_time_step'] * self.mpc.B_lip
        d = np.zeros(9)
        d[7] = - self.params['world_time_step'] * self.params['g']
        H = np.identity(3)
        Q = block_diag(1., 1., 1.)
        R = block_diag(1e1, 1e2, 1e4)
        P = np.identity(3)
        x = np.array([self.initial['com']['pos'][0], self.initial['com']['vel'][0], self.initial['zmp']['pos'][0], \
                      self.initial['com']['pos'][1], self.initial['com']['vel'][1], self.initial['zmp']['pos'][1], \
                      self.initial['com']['pos'][2], self.initial['com']['vel'][2], self.initial['zmp']['pos'][2]])
        self.kf = filter.KalmanFilter(block_diag(A, A, A), \
                                      block_diag(B, B, B), \
                                      d, \
                                      block_diag(H, H, H), \
                                      block_diag(Q, Q, Q), \
                                      block_diag(R, R, R), \
                                      block_diag(P, P, P), \
                                      x)

        # initialize residual 
        self.residual = residual_dynamics(time = 0, 
                                          starting_x=np.array([x[0], x[1], x[3], x[4], x[6], x[7]]),
                                          starting_u=np.array([x[2], x[5], x[8]]),
                                          etah=self.params['eta'], 
                                          g = self.params['g'])

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.RenderFootsteps()
        
    def customPreStep(self):
        # create current and desired states
        self.current = self.retrieve_state()

        # update kalman filter
        u = np.array([self.desired['zmp']['vel'][0], self.desired['zmp']['vel'][1], self.desired['zmp']['vel'][2]])
        self.kf.predict(u)
        x_flt, _ = self.kf.update(np.array([self.current['com']['pos'][0], self.current['com']['vel'][0], self.current['zmp']['pos'][0], \
                                            self.current['com']['pos'][1], self.current['com']['vel'][1], self.current['zmp']['pos'][1], \
                                            self.current['com']['pos'][2], self.current['com']['vel'][2], self.current['zmp']['pos'][2]]))
        
        #update residual signal
        if self.footstep_planner.get_phase_at_time(self.time) == 'ss':
            if self.footstep_planner.get_current_footstep_from_plan(self.time)['foot_id'] == self.footstep_planner.get_current_footstep_from_plan(self.residual.time)['foot_id']:
                self.residual.update(x = np.array([self.current['com']['pos'][0], self.current['com']['vel'][0],
                                                   self.current['com']['pos'][1], self.current['com']['vel'][1],
                                                   self.current['com']['pos'][2], self.current['com']['vel'][2]]), 
                                     u = np.array([self.current['zmp']['pos'][0], self.current['zmp']['pos'][1], self.current['zmp']['pos'][2]]),
                                     t = self.time)
            else:
                self.residual = residual_dynamics(time = self.time, 
                                                  starting_x = np.array([x_flt[0], x_flt[1], x_flt[3], x_flt[4], x_flt[6], x_flt[7]]),
                                                  starting_u = np.array([x_flt[2], x_flt[5], x_flt[8]]),
                                                  etah=self.params['eta'],
                                                  g = self.params['g'])

            
        # update current state using kalman filter output
        self.current['com']['pos'][0] = x_flt[0]
        self.current['com']['vel'][0] = x_flt[1]
        self.current['zmp']['pos'][0] = x_flt[2]
        self.current['com']['pos'][1] = x_flt[3]
        self.current['com']['vel'][1] = x_flt[4]
        self.current['zmp']['pos'][1] = x_flt[5]
        self.current['com']['pos'][2] = x_flt[6]
        self.current['com']['vel'][2] = x_flt[7]
        self.current['zmp']['pos'][2] = x_flt[8]

        # get references using mpc
        lip_state, contact = self.mpc.solve(self.current, self.time)

        self.desired['com']['pos'] = lip_state['com']['pos']
        self.desired['com']['vel'] = lip_state['com']['vel']
        self.desired['com']['acc'] = lip_state['com']['acc']
        self.desired['zmp']['pos'] = lip_state['zmp']['pos']
        self.desired['zmp']['vel'] = lip_state['zmp']['vel']

        if self.get_ref:
            # compute desired angular momentum about the support foot
            plan = self.footstep_planner.plan
            step_index = self.footstep_planner.get_step_index_at_time(self.time)
            support_foot_str = plan[step_index]['foot_id']

            pivot = self.lsole if support_foot_str== 'lfoot' else self.rsole
            L_des = self.compute_angular_momentum(pivot.getTransform(pivot).translation())
            self.ref_L.append(L_des)
        
        
        # get foot trajectories
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(self.time)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        # set torso and base references to the average of the feet
        for link in ['torso', 'base']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[link][key] = (self.desired['lfoot'][key][:3] + self.desired['rfoot'][key][:3]) / 2.

        # get torque commands using inverse dynamics
        commands = self.id.get_joint_torques(self.desired, self.current, contact) 
        
        # set acceleration commands
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        feet_colliding = self.feet_coll_group.collide(self.feet_coll_option, self.feet_coll_result)
        if feet_colliding:
            raise Exception("'feet_collision'")

        # log and plot
        self.logger.log_data(self.current, self.desired)
        # self.logger.update_plot(self.time)

        self.time += 1
        self.RenderFootsteps()

    def compute_angular_momentum(self, pivot=np.zeros(3)):
        L = np.zeros(3)
        for b in self.hrp4.getBodyNodes():
            L += b.getAngularMomentum(pivot)
        return L

    def retrieve_state(self, frame=dart.dynamics.Frame.World()):
        # com and torso pose (orientation and position)
        com_position = self.hrp4.getCOM(withRespectTo=frame)
        torso_orientation = get_rotvec(self.hrp4.getBodyNode('torso').getTransform(withRespectTo=frame, inCoordinatesOf=frame).rotation())
        base_orientation  = get_rotvec(self.hrp4.getBodyNode('body' ).getTransform(withRespectTo=frame, inCoordinatesOf=frame).rotation())

        # feet poses (orientation and position)
        l_foot_transform = self.lsole.getTransform(withRespectTo=frame, inCoordinatesOf=frame)
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        r_foot_transform = self.rsole.getTransform(withRespectTo=frame, inCoordinatesOf=frame)
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=frame, inCoordinatesOf=frame)
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=frame, inCoordinatesOf=frame)
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=frame, inCoordinatesOf=frame)
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=frame, inCoordinatesOf=frame)
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=frame, inCoordinatesOf=frame)

        # compute total contact force
        force = np.zeros(3)
        for contact in self.world.getLastCollisionResult().getContacts():
            force += contact.force

        # compute zmp
        zmp = np.zeros(3)
        zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / self.params['h'])
        for contact in self.world.getLastCollisionResult().getContacts():
            if contact.force[2] <= 0.1: continue
            zmp[0] += (contact.point[0] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
            zmp[1] += (contact.point[1] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])

        if force[2] <= 0.1: # threshold for when we lose contact
            zmp = np.array([0., 0., 0.]) # FIXME: this should return previous measurement
        else:
            # sometimes we get contact points that dont make sense, so we clip the ZMP close to the robot
            midpoint = (l_foot_position + l_foot_position) / 2.
            zmp[0] = np.clip(zmp[0], midpoint[0] - 0.3, midpoint[0] + 0.3)
            zmp[1] = np.clip(zmp[1], midpoint[1] - 0.3, midpoint[1] + 0.3)
            zmp[2] = np.clip(zmp[2], midpoint[2] - 0.3, midpoint[2] + 0.3)

        # create state dict
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_orientation,
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_orientation,
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3)},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)}
        }
    
    def RenderFootsteps(self):
        '''
        Method for render the footsteps planned. In red the one that are planned in the future and in blue the one that
        are already passed
        '''
        # start from the current footstep
        starting_index = self.footstep_planner.get_step_index_at_time(self.time)

        # from the last footstep up to the current remove for redraw
        for i in reversed(range(max(starting_index- 1, 0), len(self.plan_skeleton))):
            self.world.removeSkeleton(self.plan_skeleton[i])
            self.plan_skeleton.pop(i)

        # from the current footstep to the last re create rectangle
        for i in range(max(starting_index - 1, 0), len(self.footstep_planner.plan)):
            step_skel = dart.dynamics.Skeleton(f"step_{i}")
            step_skel.setGravity([0.0, 0.0, 0.0]) 
            step_skel.setMobile(False)

            joint, body = step_skel.createFreeJointAndBodyNodePair()
            
            joint.setName(f"step_{i}_joint")  
            body.setName(f"step_{i}_body")  
            # Create box shape  
            shape = dart.dynamics.BoxShape([self.params['foot_size']*2.1, self.params['foot_size']*1.5, 0.0001])  
            
            # Create shape node with visual, collision, and dynamics aspects  
            shape_node = body.createShapeNode(shape)  
            
            # Create aspects separately  
            visual = shape_node.createVisualAspect()  
            #collision = shape_node.createCollisionAspect()  
            #dynamics = shape_node.createDynamicsAspect()  
            
            # Set visual properties  
            if i >= self.footstep_planner.get_step_index_at_time(self.time): 
                visual.setColor([1.0, 0.0, 0.0, 1.0]) # red
            else: 
                visual.setColor([0.0, 0.0, 1.0, 1.0]) # blue
                        
            # Set initial position  
            transform = dart.math.Isometry3()  
            pos = self.footstep_planner.plan[i]['pos']
            ang = self.footstep_planner.plan[i]['ang']
            transform.set_translation([pos[0], pos[1], pos[2] +0.002])  
            
            # set initial rotation
            transform.set_quaternion(dart.math.Quaternion(Euler2Quaternion(ang)))

            joint.setTransform(transform)  
            
            self.world.addSkeleton(step_skel)
            self.plan_skeleton.append(step_skel)
            
    def set_get_ref_node(self, value: bool):
        self.get_ref = value
        return

def simulation_setup(render = True, angle_x = 0.0, angle_y = 0.0, trajectory=-1, get_reference = False):
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    
    # modified gravity
    g = 9.81
    gravity = [-g * np.sin(angle_y), g* np.cos(angle_y) * np.sin(angle_x), -g*np.cos(angle_x)*np.cos(angle_y)]
    world.setGravity(gravity)
    
    world.setTimeStep(0.01)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4, trajectory=trajectory, get_ref = get_reference)

    if not render: return world, None, node

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    #viewer.frame()

    return world, viewer, node



if __name__ == "__main__":
    render = True
    world, viewer, node = simulation_setup(render=render, trajectory=2, angle_x=0.06, angle_y=0.01)
    node.setTargetRealTimeFactor(10) # speed up the visualization by 10x

    # print(node.footstep_planner.plan)

    if render:
        try: # ugly: catch footstep generation continuing beyond plan's end
            viewer.run()
        except Exception as e:
            print(e)
    else:
        num_steps = 10000
        try: # ugly: catch footstep generation continuing beyond plan's end
            for step in range(num_steps):
                node.customPreStep()
                world.step()
                # if step % 5 == 0: 
                #     viewer.frame()
        except Exception as e:
            print(e)

    # node.logger.update_plot(node.time)
    input()


