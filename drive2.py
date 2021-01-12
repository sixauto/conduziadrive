import gym
gym.logger.set_level(32)
import numpy as np
from stable_baselines3 import PPO
import pickle

def load_vars():
    with open('./objs.pkl', 'rb') as f:
        rewards, steps = pickle.load(f)
        return rewards, steps

def run():
    env = gym.make('CarRacing-v0')
    drive = PPO('CnnPolicy', env, verbose=1)
    drive.learn(total_timesteps=100000)
    drive.save("conduziadrive")

    del drive

    drive = PPO.load("conduziadrive")

    obs = env.reset()
    running_score = 0
    steps = 0
    score = 0
    reward_threshold = 900
    rewards_steps = {}
    while True:
        action, _states = drive.predict(obs)
        obs, reward, done, info = env.step(action * np.array([2., 1., 1.]) + np.array([-1., -1., 0.]))
        score += reward
        steps += 1
        rewards_steps[steps] = reward
        env.render()
        running_score = running_score * 0.99 + score * 0.01
        if running_score > reward_threshold:
            print("Solved! Reward: {} and the last episode: {}!".format(running_score, score))
            drive.save("conduziadrivefinal")
            env.close()
            break
    with open('./objs.pkl', 'wb') as f:
        pickle.dump(rewards_steps, f)
run()
