import gym
import gym_compete
#import pickle
#import sys
#import argparse
import tensorflow as tf
import numpy as np

env = gym.make('sumo-ants-v0')



action_space = env.action_space.spaces
print('high: ')
print(action_space[0].high)

print('low: ' )
print(action_space[0].low)

observation_space = env.observation_space.spaces
print(observation_space)
print('high: ')
print(observation_space[0].high)

print('low: ' )
print(observation_space[0].low)

print('sample')


print(env.reset())
print(env.reset())
env.render()
print(env.action_space.sample())

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
