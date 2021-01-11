import gym
gym.logger.set_level(32)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:

def load_vars():
# Getting back the objects:
    with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
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

    total_reward = 0.0
    steps = 0
    obs = env.reset()
    rewards_steps = {}
    while True:
        action, _states = drive.predict(obs)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        steps += 1
        rewards_steps[steps] = rewards
        env.render()
    with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump(rewards_steps, f)

run()
