import MyWrapper

from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize, VecFrameStack
from MyPolicy import NoBiasActionBiasACPolicy
import time
import numpy as np
import torch
import MyPlotter
from stable_baselines3.common.callbacks import BaseCallback
class SaveOnLevelling_Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, vec_env, f_model, f_env, verbose: int = 0):
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
        self.f_model = f_model
        self.f_env = f_env
        self.vec_env = vec_env

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.training_env.model = self.model
        self.training_env.vec_env = self.vec_env
        self.training_env.file_model = self.f_model
        self.training_env.file_env = self.f_env

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


def start_simulation(model_path: str, vecnorm_path: str):
    env = MyWrapper.ISMPC2gym_env_wrapper(verbose=False,render=True,frequency_change_grav=1)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env, device="cpu")
    s, info = env.reset()
    for _ in range(1500):
        action, _states = model.predict(s, deterministic=True)
        s, r, term, trunc, info = env.step(action)
        if term or trunc: break

    env.UpdatePlot()
    input("finished")
    
def main(n_envs = 1, train = True, load = False, filename_model=f"ppo_hrp4_multienv.zip", filename_env=f"vec_normalized.pkl", desired_trajectory=100, footstep_scaler=0.9, catch_reference=False, action_decision = False, curriculum_learning=False) -> None:
    
    vec_env = make_vec_env(MyWrapper.ISMPC2gym_env_wrapper, n_envs, env_kwargs={"verbose": False, "render": n_envs == 1, "frequency_change_grav" : 1, "desired_trajectory": desired_trajectory, "curriculum_learning": curriculum_learning, "footstep_scaler": footstep_scaler, "get_L_reference" : catch_reference,  "get_ref_node" : catch_reference, "action_decision": action_decision}, vec_env_cls=SubprocVecEnv)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # for new environment
    if load:
        vec_env = VecNormalize.load(filename_env, vec_env)
        model = PPO.load(filename_model, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=1000.0, clip_reward=50_000)
        model = PPO(NoBiasActionBiasACPolicy, vec_env, verbose=1, device="cpu", n_steps=64, ent_coef=0.01, learning_rate=1e-3, n_epochs=2)
        
    if catch_reference:
        
        base_env = vec_env.venv
        
        for env_i in range(n_envs):
            
            base_env.env_method("set_disturbances",0.0, 0.0,indices=env_i)  # to disable disturbances when getting reference L
            base_env.env_method("set_Ldes", True, indices=env_i)  # to enable getting reference L in env during training
            base_env.set_attr("get_ref_node",True,indices=env_i)  # to enable getting reference L into self.node

        print("\nCatching reference for Angular Momentum\n")
        vec_env.training = False
        vec_env.norm_reward = False
        
        for i in range(1):

            s = vec_env.reset()
            done = False
            
            while not done:
                
                action = np.array([[0.0, 0.0, 0.0]]) # send action just to make the robot going forward
                s, _, done, info = vec_env.step(action)
        
        for env_i in range(n_envs):
            
            base_env.env_method("compute_Ldes", indices=env_i)
            base_env.env_method("set_Ldes", False, indices=env_i) # to disable getting reference L during training
            base_env.env_method("set_disturbances", 1.0, 1.0, indices=env_i)  # to enable disturbances during training
           # base_env.env_method("set_get_ref", False, indices=env_i)   # to disable getting reference L into self.node
        
        
        print("Getted all desired L\n")
        
    if train:
        vec_env.training = True
        vec_env.norm_reward = True
        print("start training")
        new_logger = configure('./multi.log', ["stdout", "csv", "log"])
        model.set_logger(new_logger)

        try:
            for i in range(1000):
                model.learn(total_timesteps=2048*8, callback=SaveOnLevelling_Callback(vec_env = vec_env, f_model = filename_model, f_env = filename_env))
                model.save(filename_model)
                vec_env.save(filename_env)
                print(f"last save: ppo_hrp4_multienv" + " @"*20)
        except KeyboardInterrupt:
            model.save(filename_model)
            vec_env.save(filename_env)
            print(f"last save: ppo_hrp4_multienv" + " @"*20)
        print("end training")

    start_simulation(model_path=filename_model,vecnorm_path=filename_env)    

if __name__ == "__main__":
    main(train=True, load=False, filename_model=f"ppo_hrp4_nextfootstep_curriculum2.zip", filename_env=f"ppo_hrp4_nextfootstep_curriculum2.pkl", curriculum_learning=True, n_envs=16, footstep_scaler=1., desired_trajectory=1)   