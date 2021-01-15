import gym
gym.logger.set_level(32)
import torch as th
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def train():
    #Recebe e cria o ambiente
    env = make_vec_env('CarRacing-v0', n_envs=2)
    #Cria o agent
    drive = PPO(MlpPolicy, env, gamma=.95, gae_lambda=0.8, batch_size=128, learning_rate=0.0001, clip_range=0.2, n_steps=32, n_epochs=20, verbose=1)
    # Treina o agent
    drive.learn(total_timesteps=2000)
    # Salva o treino feito pelo agent
    drive.save("conduziadrive.pkl")

    del drive
    # Faz load automatico dos argumentos
    drive = PPO.load("conduziadrive.pkl")

    # Execulta o agent 
    running_score = 0
    score = 0
    obs = env.reset()
    while True:
        for t in range(1000):
            action, _states = drive.predict(obs)
            s, r, done, info = env.step(action)
            score += r
            env.render()
        running_score = running_score * 0.99 + score * 0.01
        print (running_score)
        env.reset()

train()
