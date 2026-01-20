import MyWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from MyPolicy import NoBiasActionBiasACPolicy
from stable_baselines3.common.vec_env import  VecFrameStack
from stable_baselines3.common.logger import configure
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
    
    env = MyWrapper.ISMPC2gym_env_wrapper(verbose=False, render=True, max_step=500, frequency_change_grav=1)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.0)
    
    #model = PPO(NoBiasActionBiasACPolicy, env, verbose=1, device="cpu", n_steps=64, ent_coef=0.01, learning_rate=1e-3, n_epochs=2)
    
    #model = PPO("MlpPolicy", env, verbose=2, n_steps=128, n_epochs=3, ent_coef=0.01, learning_rate=1e-3)
    env = VecNormalize.load("vec_normalized_good_training.pkl", env)
    model = PPO.load("ppo_hrp4_multienv_good_training", env)
    #print("start training")
    #new_logger = configure('./multi.log', ["stdout", "json", "log", "tensorboard"])
    #model.set_logger(new_logger)

    '''
    for _ in range(10):
        model.learn(total_timesteps=1024)  
        model.save('ppo_hrp4')
        env.save("env_normalized.pkl")
        print('saved' + ' @'*20)
    '''
    
    print("start simulations")
    env.training = False
    env.norm_reward = False
    for i in range(1):
        print(f"simulation #{i}")
        s = env.reset()

        for _ in range(1500):
            action, _states = model.predict(s, deterministic=False)
            #action = np.array([[0.0, 0.0, 0.0]]) # send action just to make the robot going forward
            s, r, done, info = env.step(action)

            if done: 
                print("brak")
                break

        #env.UpdatePlot()

    input("finished")



if __name__ == "__main__":

    main()
    #SB3_test() # test for stable baseline 3
