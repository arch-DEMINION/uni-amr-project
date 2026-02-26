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
import test_gravity
import test_forces

def main(train = False, load = False, custom_action = True, filename_model="", filename_env="", footstep_scaler=0.9, desired_trajectory=100) -> None:
    
    env = MyWrapper.ISMPC2gym_env_wrapper(verbose=False, render=True, max_step=500, frequency_change_grav=1, footstep_scaler=footstep_scaler, desired_trajectory=desired_trajectory, test_forces=[], test_gravity=test_gravity.gravity)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    if not load:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=500)
        model = PPO(NoBiasActionBiasACPolicy, env, verbose=1, device="cpu", n_steps=64000, ent_coef=0.01, learning_rate=0., n_epochs=2)
    else:
        env = VecNormalize.load(filename_env, env)
        model = PPO.load(filename_model, env)

    new_logger = configure('./single.log', ["stdout", "log", "csv"])
    model.set_logger(new_logger)

    # model.set_logger(Logger('./single.log', [MyLogger.BarOutputFormat("test.bsv")]))

    if train:
        print("start training")
        for _ in range(10):
            # model.learn(total_timesteps=102400, callback=MyPlotter.PlotCallback(folder=f"nominal_plots_traj{desired_trajectory}_gravity_horizon100_noforces"))
            # model.learn(total_timesteps=102400, callback=MyPlotter.PlotCallback(folder=f"{filename_model}_plots_traj{desired_trajectory}_gravity_horizon150_inclined"))
            # model.learn(total_timesteps=1024)
            # model.save('ppo_hrp4')
            # env.save("env_normalized.pkl")
            print('saved' + ' @'*20)
    
    
    print("start simulations")
    env.training = False
    env.norm_reward = False
    for i in range(1):
        print(f"simulation #{i}")
        s = env.reset()

        for _ in range(50000):
            
            if custom_action: action = np.array([[0.0, 0.0, 0.0]]) # send action just to make the robot going forward
            else: action, _states = model.predict(s, deterministic=False)
            s, r, done, info = env.step(action)

            #if done: break

        #env.UpdatePlot()

    input("finished")



if __name__ == "__main__":

    main(train = False, load = True, custom_action = True, filename_model="models/ppo_hrp4_scaled09_curriculum2_footdistance2.zip", filename_env="models/ppo_hrp4_scaled09_curriculum2_footdistance2.pkl", footstep_scaler=0.9, desired_trajectory=2)
