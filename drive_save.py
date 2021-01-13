import gym
gym.logger.set_level(32)
from stable_baselines3 import PPO

def run():
    env = gym.make('CarRacing-v0')
    drive = PPO('CnnPolicy', env, verbose=1)
    drive.learn(total_timesteps=10000)
    drive.save("conduziadrive.pkl")
run()
