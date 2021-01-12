import gym
gym.logger.set_level(32)
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

import pickle

def load_vars():
    with open('objs.pkl', 'rb') as f:
        rewards, steps = pickle.load(f)
        return rewards, steps

def run():
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/')
    event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    env = make_vec_env('CarRacing-v0', n_envs=1)
    drive = PPO('MlpPolicy', env, verbose=1)

    drive.learn(int(2e4), callback=event_callback)
    drive.save("conduziadrive")

    del drive

    drive = PPO.load("conduziadrive")

    score = 0
    steps = 0
    obs = env.reset()
    running_score = 0
    reward_threshold = 900
    rewards_steps = {}
    while True:
        action, _states = drive.predict(obs)
        obs, reward, dones, info = env.step(action * np.array([2., 1., 1.]) + np.array([0., 0., 0.]))
        score += reward
        steps += 1
        rewards_steps[steps] = reward
        #env.render()
        running_score = running_score * 0.99 + score * 0.01
        print (running_score)
        if running_score > reward_threshold:
            print("Solved! Reward: {} and the last episode: {}!".format(running_score, score))
            env.close()
            break
    with open('objs.pkl', 'wb') as f:
        pickle.dump(rewards_steps, f)

run()
