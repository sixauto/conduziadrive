import gym
gym.logger.set_level(40)

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

env = make_vec_env('CarRacing-v0', n_envs=2)
drive = PPO2(MlpPolicy, env, verbose=1)
drive.learn(total_timesteps=125, log_interval=1, tb_log_name='PPO2', reset_num_timesteps=True)
drive.save("conduziadrive")

#del drive

drive = PPO2.load("conduziadrive")

obs = env.reset()
while True:
    action, _states = drive.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()