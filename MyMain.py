import MyWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from MyPolicy import NoBiasActionBiasACPolicy
from stable_baselines3.common.vec_env import  VecFrameStack
import gymnasium as gym
import numpy as np


def SB3_test() -> None:
    env = gym.make("InvertedPendulum-v5")

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)

    env_sim = gym.make("InvertedPendulum-v5", render_mode='human')
    s,_ = env_sim.reset()
    total_rew = 0
    tranchs = 1

    for i in range(1_000):
        action, _states = model.predict(s, deterministic=True)
        s, r, term, trunc, _ = env_sim.step(action)

        if term or trunc: 
            s, _ = env_sim.reset()
            tranchs += 1

        total_rew += r

    env.close()
    env_sim.close()
    print(total_rew/tranchs)


def main() -> None:
    
    env = MyWrapper.ISMPC2gym_env_wrapper(verbose=False, render=True, max_step=1_000)
    env = DummyVecEnv([lambda: env])
    
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load("ppo_hrp4_multienv1.zip", env=env)
    
    # for _ in range(10):
    #     model.learn(total_timesteps=1024)
    #     print('saved')
    
    print("start simulations")
    s = env.reset()
    for _ in range(150000):
        action, _states = model.predict(s, deterministic=False)
        s,r,done,info = env.step(action)

    input("finished")



if __name__ == "__main__":

    main()
    #SB3_test() # test for stable baseline 3
