import gym
gym.logger.set_level(32)
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def train():
    #Recebe e cria o ambiente
    env = make_vec_env ('CarRacing-v0', n_envs=6)
    #Cria o agent
    #drive = PPO(MlpPolicy, env, gamma=0.9997, gae_lambda=1, ent_coef=0.01, vf_coef=1, batch_size=4096, learning_rate=5e-6, clip_range=0.1, n_steps=5000, n_epochs=100, target_kl=0.03, verbose=1)
    # Treina o agent
    #drive = drive.learn(total_timesteps=250000, log_interval=10).save("conduziadrive")

    #del drive
    # Faz load automatico dos argumentos
    drive = PPO.load("conduziadrive")
    # Executa o agent
    rewards = []
    obs = env.reset()
    while True:
        total_rewards = 0
        obs = env.reset()
        for t in range(1000):
            action, _states = drive.predict(obs)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
            #env.render()
            if(t % 100 == 0):
                print(t)
            if done.any or t == 999:
                print("Episode {} finished after {} timesteps".format(5, t+1))
                print("Reward: {}".format(total_rewards))
                np.append(rewards, total_rewards)
            if done.any:
                break

train()
