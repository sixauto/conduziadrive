import gym
gym.logger.set_level(32)

import pybullet
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train():
    # Get the enviroment from GYM
    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10., epsilon=0.1)
    # Create the Agent for training
    drive = PPO(MlpPolicy, env, gamma=0.9997, ent_coef=0.01, vf_coef=1, batch_size=128, learning_rate=linear_schedule(0.001), clip_range=linear_schedule(0.1), n_steps=1000, n_epochs=20, verbose=1)
    # Training the Agent
    drive.learn(total_timesteps=50000)
    # Save the Training
    drive.save("conduziadrive")
    env.save("conduziadrive.pkl")
    env.close()

def run():
    # Load the traininig
    drive = PPO.load("conduziadrive")
    # Get the enviroment and normalize this enviroment for execution
    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    env = VecNormalize.load("conduziadrive.pkl", env)
    #  do not update them at test time
    env.training = False
    # Reset the enviroment
    obs = env.reset()
    while True:
        obs = env.reset()
        total_rewards = 0
        for t in range(1000):
            # Receive the action and states of the agent trained before
            action, _states = drive.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
            # Rendering the enviroment to show this in graph scale
            env.render()
            # Validate if the agent get the "goal" or "die" after a failed round
            if done:
                print("Episode {} finished after {} timesteps".format(5, t+1))
                print("Reward: {}".format(total_rewards))
    env.close()

train()