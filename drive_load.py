import gym
gym.logger.set_level(32)
import numpy as np
from stable_baselines3 import PPO

def run():
    env = gym.make('CarRacing-v0')
   
    drive = PPO('CnnPolicy', env, verbose=1)
    drive = PPO.load("conduziadrive.pkl")

    running_score = 0
    score = 0
    reward_threshold = 900
    obs = env.reset()
    while True:
        action, _states = drive.predict(obs)
        obs, reward, done, info = env.step(action * np.array([1., 1., 1.]) + np.array([0., 0., 0.]))
        score += reward
        env.render()
        print (score, action)
        if done:
            env.reset()
        running_score = running_score * 0.99 + score * 0.01
        if running_score > reward_threshold:
            print("Solved! Reward: {} and the last episode: {}!".format(running_score, score))
            drive.save("conduziadrivefinal.pkl")
            env.close()
            break
run()
