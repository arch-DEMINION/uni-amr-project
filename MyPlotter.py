from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import *
import utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

class PlotCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0, folder="plots"):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.folder = folder

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # since we are using vectorized environments, variables have to be retrieved like this
        # this returns an array of all the variables in each environment
        
        ep_step = self.training_env.get_attr("current_step")[0]
        episode = self.training_env.get_attr("episodes")[0]

        start_step = 2 if episode == 1 else 1
        
        # ok this works, but is inefficient as hell
        # if ep_step > start_step:
        #     self.logger.record("mpc_state", self._state)
        #     self.logger.record("timestep", self._ep_step)
        #     self.logger.record("plan", self._plan)
        #     self.logger.record("original_plan", self._original_plan)
        #     self.logger.record("episode", self._episode)
            
        #     self.logger.dump()

        # self._plan  = self.training_env.get_attr("plan")[0]
        # self._original_plan  = self.training_env.get_attr("original_plan")[0]
        # self._state = str(self.training_env.get_attr("mpc_state")[0]).replace("\n", "") #strip stray newlines, I don't know where they come from but the completely mess up the formatting
        # self._ep_step = ep_step
        # self._episode = episode

        # the callback is not called if the episode is terminated or truncated (like what? why??)
        # so delay everything by one step and catch when a new episode starts        
        # step 0 is the reset, but gets counted only after a termination or truncation, for some reason
        # for the first iteration , delay everything by one step

        # from the second episode on, save the last plan of the previous episode
        if ep_step == 0 and episode > 1:
            # self.logger.record("episode", self._episode)
            # self.logger.record("timestep", self._ep_step)
            # self.logger.record("original_plan", None) # empty entry, for proper formatting
            # self.logger.record("mpc_state", self._state)
            # self.logger.record("plan", self._plan)
            # # actually write to file
            # self.logger.dump()

            fig, ax = plt.subplots()
            plt.title("Plan")
            plt.xlabel("x")
            plt.ylabel("y")

            foot_size=0.1
            margin = 2*foot_size
            offset = np.array([foot_size, foot_size])

            xrange = self.maxx - self.minx
            xmid = (self.maxx+self.minx)*0.5
            yrange = self.maxy - self.miny
            ymid = (self.maxy+self.miny)*0.5

            if xrange >= yrange:
                ax.set_xlim(self.minx-margin, self.maxx+margin)
                ax.set_ylim(ymid-xrange*0.5-margin, ymid+xrange*0.5+margin)
            else:
                ax.set_ylim(self.miny-margin, self.maxy+margin)
                ax.set_xlim(xmid-yrange*0.5-margin, xmid+yrange*0.5+margin)

            # ax.set_xlim(-foot_size*2, 1.2)
            # # ax.set_xlim(-0.6-foot_size, 0.6+foot_size)
            # ax.set_ylim(-0.6-foot_size, 0.6+foot_size)

            # plot footsteps, red for left, blue for right
            for s in self._plan:
                r = patches.Rectangle(
                    s['pos'][:2] - offset, #(left, bottom), width height
                    2*foot_size, 2*foot_size,
                    rotation_point='center',
                    angle=s['ang'][2],
                    fill=False, lw=2.5,
                    color='r' if s['foot_id']=='lfoot' else 'b'
                )
                ax.add_patch(r)
                
            # plot original footsteps, red for left, blue for right
            for s in self._original_plan:
                r = patches.Rectangle(
                    s['pos'][:2] - offset, #(left, bottom), width height
                    2*foot_size, 2*foot_size,
                    rotation_point='center',
                    angle=s['ang'][2],
                    fill=True, lw=1,
                    color='r' if s['foot_id']=='lfoot' else 'b',
                    linestyle='--', alpha=0.15
                )
                ax.add_patch(r)
            
            # plot com
            zx = []
            zy = []
            for s in self._states:
                zx.append(s['com']['pos'][0])
                zy.append(s['com']['pos'][1])

            ax.plot(zx, zy, 'g--', linewidth=2 )

            # plot disturbance forces, as seen force above
            for f in self._forces:
                x = f['arrow_tail'][0]
                y = f['arrow_tail'][1]
                dx = f['arrow_head'][0]-f['arrow_tail'][0]
                dy = f['arrow_head'][1]-f['arrow_tail'][1]
                p = patches.Arrow(x,y,dx,dy,width=0.12)
                ax.add_patch(p)

            # plot gravity vectors, as seen from above, slightly below the com position when the change happened
            # offset = -1.0
            # for f in self._gravities:
            #     x = f['com_pos_at_change'][0]
            #     y = f['com_pos_at_change'][1]
            #     dx = f['gravity'][0]
            #     dy = f['gravity'][1]
            #     p = patches.Arrow(x,y+offset,dx,dy,width=0.12)
            #     ax.add_patch(p)

            ax.grid(visible=True, which='both', linewidth=1, alpha=0.3)
            # show
            # plt.show()
            fig.savefig(f"{self.folder}/episode{self._episode}.pdf", format="pdf", )

            # clear stores states for next episode
            self._states = []
        else:
            # any episode, after one step as passed
            if ep_step >= start_step:
                # # always save episode and timestep
                # self.logger.record("episode", self._episode)
                # self.logger.record("timestep", self._ep_step)

                # # if first step of new episode, save the original plan
                # if ep_step == start_step:
                #     original_plan = self.training_env.get_attr("original_plan")[0]
                #     self.logger.record("original_plan", original_plan)

                # # save mpc state. Leave plan entry empty for proper formatting
                # self.logger.record("mpc_state", self._state)
                # self.logger.record("plan", None) # empty entry, for proper formatting
                # # actually write to file
                # self.logger.dump()
                pass
            else:
                self._states = []

                self.minx = 0
                self.miny = 0
                self.maxx = 0
                self.maxy = 0

        # save plan and state of this step. Will be logged next step
        self._plan  = self.training_env.get_attr("plan")[0]
        self._original_plan  = self.training_env.get_attr("original_plan")[0]
        self._state = self.training_env.get_attr("mpc_state")[0]
        self._ep_step = ep_step
        self._episode = episode
        self._forces = self.training_env.get_attr("forces")[0]
        self._gravities = self.training_env.get_attr("gravities")[0]

        self._states.append(self._state)

        pos = self._state['com']['pos']
        if pos[0] < self.minx: self.minx=pos[0]
        if pos[1] < self.miny: self.miny=pos[1]
        if pos[0] > self.maxx: self.maxx=pos[0]
        if pos[1] > self.maxy: self.maxy=pos[1]

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
