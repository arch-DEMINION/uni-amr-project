import MyWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from MyPolicy import NoBiasActionBiasACPolicy
from stable_baselines3.common.vec_env import  VecFrameStack
from stable_baselines3.common.logger import configure
import gymnasium as gym
import numpy as np
import os
from stable_baselines3.common.logger import Logger
import MyLogger
import MyPlotter

def main(train = False, load = False, custom_action = True) -> None:
    
    env = MyWrapper.ISMPC2gym_env_wrapper(verbose=False, render=True, max_step=500, frequency_change_grav=1, footstep_scaler=0., desired_trajectory=101)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    if not load:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=500)
        model = PPO(NoBiasActionBiasACPolicy, env, verbose=1, device="cpu", n_steps=64, ent_coef=0.01, learning_rate=1e-3, n_epochs=2)
    else:
        env = VecNormalize.load("ppo_hrp4_nextfootstep_curriculum.pkl", env)
        model = PPO.load("ppo_hrp4_nextfootstep_curriculum.zip", env)

    new_logger = configure('./single.log', ["stdout", "log", "csv"])
    model.set_logger(new_logger)

    # model.set_logger(Logger('./single.log', [MyLogger.BarOutputFormat("test.bsv")]))

    if train:
        print("start training")
        for _ in range(10):
            model.learn(total_timesteps=1024, callback=MyPlotter.PlotCallback())  
            model.save('ppo_hrp4')
            env.save("env_normalized.pkl")
            print('saved' + ' @'*20)
    
    
    print("start simulations")
    env.training = False
    env.norm_reward = False
    for i in range(1):
        print(f"simulation #{i}")
        s = env.reset()

        for _ in range(1500):
            
            if custom_action: action = np.array([[0.0, 0.0, 0.0]]) # send action just to make the robot going forward
            else: action, _states = model.predict(s, deterministic=False)
            s, r, done, info = env.step(action)

            #if done: break

        #env.UpdatePlot()

    input("finished")



if __name__ == "__main__":

    main(train = True, load = True, custom_action = True)
