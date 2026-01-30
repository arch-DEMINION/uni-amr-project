from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import *
import utils

from matplotlib import pyplot as plt

class BarOutputFormat(KVWriter):
    """
    Log to a file, in a CSV format

    :param filename: the file to write the log to
    """

    def __init__(self, filename: str):
        self.file = open(filename, "w+")
        self.keys: list[str] = []
        self.entry_separator = "|"
        self.newline_separator = "\\"
        self.quotechar = '"'

    def write(self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0) -> None:
        # Add our current row to the history
        key_values = filter_excluded_keys(key_values, key_excluded, "csv")
        extra_keys = key_values.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for i, key in enumerate(self.keys):
                if i > 0:
                    self.file.write(self.entry_separator)
                self.file.write(key)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.newline_separator * len(extra_keys))
                self.file.write("\n")
        for i, key in enumerate(self.keys):
            if i > 0:
                self.file.write(self.entry_separator)
            value = key_values.get(key)

            if isinstance(value, Video):
                raise FormatUnsupportedError(["csv"], "video")

            elif isinstance(value, Figure):
                raise FormatUnsupportedError(["csv"], "figure")

            elif isinstance(value, Image):
                raise FormatUnsupportedError(["csv"], "image")

            elif isinstance(value, HParam):
                raise FormatUnsupportedError(["csv"], "hparam")

            elif isinstance(value, str):
                # escape quotechars by prepending them with another quotechar
                value = value.replace(self.quotechar, self.quotechar + self.quotechar)

                # additionally wrap text with quotechars so that any delimiters in the text are ignored by csv readers
                self.file.write(self.quotechar + value + self.quotechar)

            elif value is not None:
                self.file.write(str(value))
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        self.file.close()

class LogCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
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
            self.logger.record("episode", self._episode)
            self.logger.record("timestep", self._ep_step)
            self.logger.record("original_plan", None) # empty entry, for proper formatting
            self.logger.record("mpc_state", self._state)
            self.logger.record("plan", self._plan)
            # actually write to file
            self.logger.dump()
        else:
            # any episode, after one step as passed
            if ep_step >= start_step:
                # always save episode and timestep
                self.logger.record("episode", self._episode)
                self.logger.record("timestep", self._ep_step)

                # if first step of new episode, save the original plan
                if ep_step == start_step:
                    original_plan = utils.lighten_plan(self.training_env.get_attr("original_plan")[0])
                    self.logger.record("original_plan", original_plan)

                # save mpc state. Leave plan entry empty for proper formatting
                self.logger.record("mpc_state", self._state)
                self.logger.record("plan", None) # empty entry, for proper formatting
                # actually write to file
                self.logger.dump()

        # save plan and state of this step. Will be logged next step
        self._plan  = utils.lighten_plan(self.training_env.get_attr("plan")[0])
        self._state = str(self.training_env.get_attr("mpc_state")[0]).replace("\n", "")
        self._ep_step = ep_step
        self._episode = episode

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
