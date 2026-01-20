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

def main() -> None:
    
    vec_env = make_vec_env(MyWrapper.ISMPC2gym_env_wrapper, n_envs=8, env_kwargs={"verbose": False, "render": False, "frequency_change_grav" : 1}, vec_env_cls=SubprocVecEnv)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # for new wnvironment
    #vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=500)
    #model = PPO(NoBiasActionBiasACPolicy, vec_env, verbose=1, device="cpu", n_steps=64, ent_coef=0.05, learning_rate=1e-3, n_epochs=2)
    
    # for loading 
    vec_env = VecNormalize.load("vec_normalized.pkl", vec_env)
    model = PPO.load("ppo_hrp4_multienv", vec_env)
    
    print("start training")
    new_logger = configure('./multi.log', ["stdout", "json", "log", "tensorboard"])
    model.set_logger(new_logger)
    for i in range(1001):
        model.learn(total_timesteps=2048)
        model.save(f"ppo_hrp4_multienv")
        vec_env.save("vec_normalized.pkl")
        print(f"last save: ppo_hrp4_multienv{i%5}" + " @"*20)
        
        if i == 1000:
            start_simulation(model_path=f"ppo_hrp4_multienv{i%5}",vecnorm_path="vec_normalized.pkl")
    print("end training")
    

    # model = PPO.load("ppo_hrp4_multienv4_", env=env, device="cpu")
    # env.training = False
    # print("start simulations")
    # for i in range(1):
    #     print(f"simulation #{i}")
    #     s, info = env.reset()

    #     for _ in range(1500):
    #         action, _states = model.predict(s, deterministic=True)
    #         #action = np.array([0.001, 0, 0.0]) # send action just to make the robot going forward
    #         s, r, term, trunc, info = env.step(action)

    #         if term or trunc: break

    #     env.UpdatePlot()

    # input("finished")



if __name__ == "__main__":
    main()

def start_simulation(model, env):
    
     model = PPO.load("ppo_hrp4_multienv4_", env=env, device="cpu")
     env.training = False
     env.norm_reward = False
     print("start simulations")
     for i in range(1):
         print(f"simulation #{i}")
         s, info = env.reset()

         for _ in range(1500):
             action, _states = model.predict(s, deterministic=True)
             #action = np.array([0.001, 0, 0.0]) # send action just to make the robot going forward
             s, r, term, trunc, info = env.step(action)

             if term or trunc: break

         env.UpdatePlot()

     input("finished")

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
        #action = np.array([0.001, 0, 0.0]) # send action just to make the robot going forward
        s, r, term, trunc, info = env.step(action)

        if term or trunc: break

    env.UpdatePlot()

    input("finished")
    