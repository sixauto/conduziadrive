import gym
gym.logger.set_level(32)

import os
import argparse
import numpy as np
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize

parser = argparse.ArgumentParser(description='CarRacing-v0')
parser.add_argument('--train', action='store_true', help='Train the agent')
parser.add_argument('--run', action='store_true', help='Run the agent')

gym_env_id = "CarRacing-v0"
total_timesteps = 40000
total_train_runs = 60
log_dir = "logs/car_drive/monitor"
env = make_vec_env("CarRacing-v0", n_envs=1, monitor_dir=log_dir)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print(f"Saving new best drive to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train(env, log_dir):
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10., clip_reward=10., epsilon=0.1)

    drive = PPO("MlpPolicy", env, ent_coef=0.01, vf_coef=1, batch_size=32, learning_rate=linear_schedule(0.001), clip_range=linear_schedule(0.1), n_steps=1000, n_epochs=20, tensorboard_log=log_dir + "/drive_tensorboard_log", verbose=1)

    drive.learn(total_timesteps=total_timesteps, callback=callback)

    for i in range(total_train_runs):
        env.close()
        drive.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)

    drive.save("conduziadrive")


def run(env):
    drive = PPO.load("conduziadrive")

    env = VecVideoRecorder(env, log_dir + '/videos/',
                       record_video_trigger=lambda x: x == 0, video_length=1000,
                       name_prefix="conduzia-drive-agent-{}".format(gym_env_id))

    env = VecNormalize(env, gamma=0.9997, norm_obs=True, norm_reward=True, clip_obs=10., epsilon=0.1)

    rewards = []
    total_reward = 0

    while True:
        obs = env.reset()

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

args = parser.parse_args()

if args.train:
    train(env, log_dir)

if args.run:
    run(env)
