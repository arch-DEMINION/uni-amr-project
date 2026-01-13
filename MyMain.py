import MyWrapper

from stable_baselines3 import PPO
import gymnasium as gym
import time
import numpy as np
import torch 

from stable_baselines3.common.monitor import Monitor

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
    model = PPO.load("ppo_hrp4_multienv3", env=env, device="cpu", force_reset=True)

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



if __name__ == "__main__":

    main()
    #SB3_test() # test for stable baseline 3
