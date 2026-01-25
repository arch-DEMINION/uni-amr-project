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


def main(train = False, load = False, catch_reference = False, custom_action = True) -> None:
    
    env = MyWrapper.ISMPC2gym_env_wrapper(verbose=False, render=True, max_step=500, frequency_change_grav=1)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    if not load:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1000.0, clip_reward=500)
        model = PPO(NoBiasActionBiasACPolicy, env, verbose=1, device="cpu", n_steps=64, ent_coef=0.01, learning_rate=1e-3, n_epochs=2)
    else:
        env = VecNormalize.load("vec_normalized.pkl", env)
        model = PPO.load("ppo_hrp4_multienv", env)
    
    #new_logger = configure('./multi.log', ["stdout", "json", "log", "tensorboard"])
    #model.set_logger(new_logger)
    
    if catch_reference:
        
        env.envs[0].unwrapped.set_disturbances(0.0, 0.0)
        env.envs[0].unwrapped.set_Ldes(catch_reference)
        print("\nCatching reference for Angular Momentum\n")
        env.training = False
        env.norm_reward = False
        
        for i in range(1):
            
            s = env.reset()
            done = False
            
            while not done:
                
                action = np.array([[0.0, 0.0, 0.0]]) # send action just to make the robot going forward
                s, r, done, info = env.step(action)

        print("Getted all desired L\n")
        
    env,env.envs[0].unwrapped.compute_Ldes()  
    env.envs[0].unwrapped.set_Ldes(False)           # to disable getting reference L during training/simulation
    env.envs[0].unwrapped.set_disturbances(1.0, 1.0) # to able disturbances during training/simulation
        
    if train:
        #print("start training")
        env.training = True
        env.norm_reward = True
        for _ in range(10):
            model.learn(total_timesteps=1024)  
            model.save('ppo_hrp4_AM')
            env.save("env_normalized_AM.pkl")
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

    main(train = True, load = False, catch_reference = True, custom_action = False)
    #SB3_test() # test for stable baseline 3
