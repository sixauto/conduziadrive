import gym
gym.logger.set_level(32)
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def train():
    #Recebe e cria o ambiente
    env = make_vec_env('CarRacing-v0', n_envs=16)
    #Cria o agent
    drive = PPO(MlpPolicy, env, gamma=0.9997, gae_lambda=1, ent_coef=0.01,  vf_coef=0.5, batch_size=128, learning_rate=5e-6, clip_range=0.1, n_steps=128, n_epochs=20, verbose=1)
    # Treina o agent
    drive = drive.learn(total_timesteps=250000, eval_freq=200, n_eval_episodes=5, log_interval=1000).save("conduziadrive")

    #del drive
    # Faz load automatico dos argumentos
    #drive = PPO.load("conduziadrive")

    # Executa o agent
    #obs = env.reset()
    #while True:
    #    action, _states = drive.predict(obs)
    #    obs, reward, done, info = env.step(action)
    #    env.render()

train()
