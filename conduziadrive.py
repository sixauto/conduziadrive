import gym
gym.logger.set_level(32)
import numpy as np
import pybullet_envs

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def train():
    #Recebe e cria o ambiente
    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.9997, clip_obs=10.)
    #Cria o agent
    drive = PPO(MlpPolicy, env,  gae_lambda=1, ent_coef=0.01, vf_coef=1, batch_size=4096, learning_rate=5e-6, clip_range=0.1, n_steps=5000, n_epochs=100, target_kl=0.03, verbose=1)
    # Treina o agent
    drive = drive.learn(total_timesteps=500000)
    # Save the enviroment
    drive.save("conduziadrive")
    env.save("conduziadrive_normalized.pkl")

    del drive, env
    # Faz load automatico dos argumentos
    drive = PPO.load("conduziadrive")

    env = DummyVecEnv([lambda: gym.make("CarRacing-v0")])
    env = VecNormalize.load("conduziadrive_normalized.pkl", env)

    env.training = False
    env.norm_reward = False
    # Test the agent
    rewards = []
    total_rewards = 0
    obs = env.reset()
    while True:
        mean_reward, std_reward = evaluate_policy(drive, env, n_eval_episodes=10)
        obs = env.reset()
        total_rewards = 0
        for t in range(1000):
            action, _states = drive.predict(obs)
            obs, reward, done, info = env.step(action)
            total_rewards += reward
            if(t % 100 == 0):
                print(t)
            if done or t == 999:
                print("Episode {} finished after {} timesteps".format(5, t+1))
                print("Reward: {}".format(total_rewards))
                np.append(rewards, total_rewards)
                total_rewards = 0

train()
