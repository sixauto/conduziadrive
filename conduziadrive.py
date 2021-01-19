import gym
gym.logger.set_level(32)
import numpy as np
import pybullet
from typing import Callable

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train():
    #Recebe e cria o ambiente
    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10.)
    #Cria o agent
    drive = PPO(MlpPolicy, env,  gae_lambda=1, ent_coef=0.01, vf_coef=1, learning_rate=linear_schedule(0.001), clip_range=0.1, n_epochs=30, verbose=1)
    # Treina o agent
    drive.learn(total_timesteps=25000)
    # Salva o treino
    drive.save("conduziadrive")
    # Salva o ambiente normalizado
    env.save("conduziadrive_normalized.pkl")

    rewards = []
    total_rewards = 0
    obs = env.reset()
    while True:
        mean_reward, std_reward = evaluate_policy(drive, env, n_eval_episodes=5, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        obs = env.reset()
        total_rewards = 0
        for t in range(1000):
            action, _states = drive.predict(obs)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
            env.render()
            if(t % 100 == 0):
                print(t)
            if done or t == 999:
                print("Episode {} finished after {} timesteps".format(5, t+1))
                print("Reward: {}".format(total_rewards))
                np.append(rewards, total_rewards)
                total_rewards = 0

train()
