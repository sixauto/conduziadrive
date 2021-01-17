import gym
gym.logger.set_level(32)
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def train():
    #Recebe e cria o ambiente
    env = make_vec_env('CarRacing-v0', seed=0, n_envs=6)
    #Cria e Treina o Agent
    drive = PPO(MlpPolicy, env, gamma=0.9997, gae_lambda=1, ent_coef=0.01, vf_coef=1, batch_size=256, learning_rate=5e-6, clip_range=0.1, n_steps=4096, n_epochs=30).learn(total_timesteps=1600000)
    # Salva o treino feito pelo Agent
    drive.save("conduziadrive")

    #del drive
    # Faz load automatico dos argumentos antes treinados
    #drive = PPO.load("conduziadrive")

    # Executa o agent
    #score = 0
    #running_score = 0
    #observation = env.reset()
    #while True:
    #    env.render()
    #    action = drive.predict(observation)
    #    observation, reward, done, info = env.step(action)
    #    score += reward
    #    if done.all():
    #       score = 0
    #       running_score = 0
    #       observation = env.reset()
    #running_score = running_score * 0.99 + score * 0.01
    #print(running_score)
    #env.close()

train()
