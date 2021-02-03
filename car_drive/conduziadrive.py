import gym
gym.logger.set_level(40)

from datetime import datetime
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
import argparse

parser = argparse.ArgumentParser(description='CarRacing-v0')
parser.add_argument('--train', action='store_true', help='Train the agent')
parser.add_argument('--run', action='store_true', help='Run the agent')

gym_env_id = "CarRacing-v0"
gym_env_mode = "human"
env = DummyVecEnv([lambda: gym.make(gym_env_id)])
env = VecNormalize(env, gamma=0.9997, norm_obs=True, norm_reward=True, clip_obs = 10., epsilon=0.2)
drive = PPO("MlpPolicy", env, ent_coef=0.01, vf_coef=1, batch_size=125, learning_rate=0.0001, clip_range=0.1, 
            n_steps=250, n_epochs=20, tensorboard_log="conduzia_drive_tensor_log", verbose=1)

def train(env):
    total_timesteps = 40000
    total_train_runs = 60
    action_space = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, .8], [0, 0, 1]]

    drive.learn(total_timesteps=total_timesteps)

    for i in range(total_train_runs):
        env.close()
        drive.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)

    drive.save("conduzia_drive_train")


def run(env):
    drive = PPO.load("conduzia_drive_train")

    env = VecVideoRecorder(env, 'logs/videos/',
                           record_video_trigger=lambda x: x == 0, video_length=1000,
                           name_prefix="conduzia-drive-agent-{}".format(gym_env_id))

    rewards = []
    total_reward = 0

    while True:
        obs = env.reset()

        for t in range(1000):
            action, _states = drive.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render(mode=gym_env_mode)
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
    train(env)

if args.run:
    run(env)
