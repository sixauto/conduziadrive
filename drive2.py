import gym
gym.logger.set_level(40)

from stable_baselines3 import PPO
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('CarRacing-v0', n_envs=1)
drive = PPO('MlpPolicy', env, verbose=1)
drive.learn(total_timesteps=400)
drive.save("conduziadrive")

del drive

drive = PPO.load("conduziadrive")

obs = env.reset()
while True:
    action, _states = drive.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
