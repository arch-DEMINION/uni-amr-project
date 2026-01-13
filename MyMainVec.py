import MyWrapper

from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
import time
import numpy as np
import torch

def main() -> None:
    vec_env = make_vec_env(MyWrapper.ISMPC2gym_env_wrapper, n_envs=8, env_kwargs={"verbose": False, "render": False}, vec_env_cls=SubprocVecEnv)

    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", n_steps=32, ent_coef=0.05, learning_rate=1e-3, n_epochs=2)
    # trick for setting to zero last layer
    last_layer = model.policy.action_net
    # print(last_layer)
    last_layer.bias = None
    torch.nn.init.constant_(last_layer.weight, 0.0)
    model.policy.action_net = last_layer    
    
    print("start training")
    new_logger = configure('./multi.log', ["stdout", "json", "log", "tensorboard"])
    model.set_logger(new_logger)
    for i in range(1000):
        model.learn(total_timesteps=2048)
        model.save(f"ppo_hrp4_multienv{i%5}")
    print("end training")

    # model = PPO.load("ppo_hrp4_multienv4_", env=env, device="cpu")
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
    #SB3_test() # test for stable baseline 3
