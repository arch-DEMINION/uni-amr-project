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
    env = Monitor(env, filename="monitor.csv")
    #MyWrapper.ISMPC2gym_env_wrapper
    
    policy_kwargs = dict( activation_fn =torch.nn.Tanh, net_arch = dict( pi = [256, 256, 128] , vf = [256 , 256, 64]))

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=2, device="cpu", n_steps=1024, ent_coef=0.05)  # ent_coef = 0.05 nominally
    
    # trick for setting to zero last layer
    
    last_layer = model.policy.action_net
    last_layer.bias = None
    torch.nn.init.constant_(last_layer.weight, 0.0)
        
    model.policy.action_net = last_layer    
    
    #model.load("ppo_hrp4_4")

    print("start training")
    for i in range(100):
        model.learn(total_timesteps=1024, progress_bar=True)
        model.save("ppo_hrp4_5")
        print(f'saved {i}')
    print("end training")
    #env.UpdatePlot()

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
