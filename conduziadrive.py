import gym
gym.logger.set_level(32)
import numpy as np
import pybullet
from typing import Callable

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train():
    #Recebe e cria o ambiente
    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10., clip_reward=10., epsilon=0.1)
    #Cria o agent
    drive = PPO(MlpPolicy, env, ent_coef=0.01, vf_coef=1, batch_size=32, learning_rate=linear_schedule(0.001), clip_range=linear_schedule(0.1), n_steps=1000, n_epochs=20, verbose=1)
    # Treina o agent
    drive.learn(total_timesteps=25000)
    # Salva o treino
    drive.save("conduziadrive")

    del env, drive

    drive = PPO.load("conduziadrive")

    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, gamma=0.9997, clip_obs=10., epsilon=0.1)
    rewards = []
    total_rewards = 0
    obs = env.reset()
    while True:
        obs = env.reset()
        total_rewards = 0
        for t in range(1000):
            action, _states = drive.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
            env.render()
            if done:
                print("Episode {} finished after {} timesteps".format(5, t+1))
                print("Reward: {}".format(total_rewards))
                np.append(rewards, total_rewards)
                print (reward)
                total_rewards = 0

train()
