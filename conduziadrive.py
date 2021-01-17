import gym
gym.logger.set_level(32)
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def train():
    #Recebe e cria o ambiente
    env = make_vec_env('CarRacing-v0', seed=0, n_envs=1)
    #Cria o agent
    drive = PPO(MlpPolicy, env, gamma=0.9997, gae_lambda=1, ent_coef=0.01, vf_coef=1, batch_size=256, learning_rate=5e-6, clip_range=0.1, n_steps=1024, n_epochs=20)
    # Treina o agent
    drive = drive.learn(total_timesteps=500000)
    # Salva o treino feito pelo agent
    drive.save("conduziadrive")

    #del drive
    # Faz load automatico dos argumentos
    #drive = PPO.load("conduziadrive")

    # Executa o agent
    #obs = env.reset()
    #while True:
    #    score = 0
    #    running_score = 0
    #    reward_threshold = 900
    #    obs = env.reset()
    #    for t in range(1000):
    #        action, _states = drive.predict(obs)
    #        obs, reward, done, info = env.step(action)
    #        score += reward
    #        env.render()
    #        if done.any():
    #            break
    #    running_score = running_score * 0.99 + score * 0.01
    #    print(running_score)
    #    if running_score > reward_threshold:
    #        print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
    #        drive.save("conduziadrivefinal")
    #        break

train()
