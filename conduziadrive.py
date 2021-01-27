import gym
gym.logger.set_level(40)

from datetime import datetime
import pybullet
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

def virtual_display():
    display = Display(visible=False, size=(100, 60), color_depth=24)
    display.start()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train():
    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10., epsilon=0.1)

    drive = PPO(MlpPolicy, env, ent_coef=0.01, vf_coef=1, batch_size=128, learning_rate=linear_schedule(0.001),
                clip_range=linear_schedule(0.1), n_steps=1000, n_epochs=20, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='drive_checkpoint')

    drive.learn(total_timesteps=40000, callback=checkpoint_callback)
    runs = 7

    for i in range(runs):
        env.close()
        drive.learn(total_timesteps=40000, callback=checkpoint_callback, reset_num_timesteps=False)

    drive.save("drive_train_{}".format(datetime.now().strftime("%d/%m/%Y_%H:%M:%S")))

    mean_reward, std_reward = evaluate_policy(drive, env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


def run():
    drive = PPO.load("./logs/drive_checkpoint_320000_steps")

    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10., epsilon=0.1)

    mean_reward, std_reward = evaluate_policy(drive, env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    rewards = []
    total_reward = 0

    while True:
        obs = env.reset()

        mean_reward, std_reward = evaluate_policy(drive, env, n_eval_episodes=10)
        print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

        for t in range(1000):
            action, _states = drive.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
            if t % 100 == 0:
                print(t)
            if done:
                break

        
        print("Finished after {} timesteps".format(t+1))
        print("Reward: {}".format(total_reward))
        rewards.append(total_reward)
        env.close()


virtual_display()
#train()
run()
