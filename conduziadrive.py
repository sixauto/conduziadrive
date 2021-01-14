import gym
gym.logger.set_level(32)
import torch as th
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def train():
    # Recebe e cria o ambiente
    env = make_vec_env('CarRacing-v0', n_envs=8)
    # Cria o agent
    drive = PPO(MlpPolicy, env, gamma=0.98, gae_lambda=0.8, batch_size=256, clip_range=0.2, ent_coef=0.0, learning_rate=0.001, n_steps=32, n_epochs=20, verbose=1)
    # Treina o agent
    drive.learn(total_timesteps=100000)
    # Salva o treino feito pelo agent
    drive.save("conduziadrive.pkl")

    del drive
    # Faz load automatico dos argumentos
    drive = PPO.load("conduziadrive.pkl")

    # Execulta o agent 
    running_score = 0
    score = 0
    reward_threshold = 900
    obs = env.reset()
    while True:
        for t in range(1000):
            action, _states = drive.predict(obs)
            obs, reward, done, info = env.step(action * np.array([1., 1., 1.]) + np.array([0., 0., 0.]))
            score += reward
            env.render()
        print (running_score)
        running_score = running_score * 0.99 + score * 0.01
        if running_score > reward_threshold:
            print("Solved! Reward: {} and the last episode: {}!".format(running_score, score))
            env.save("conduziadrivefinal.pkl")
            env.close()

train()
