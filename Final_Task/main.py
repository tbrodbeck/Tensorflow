import gym
import tensorflow as tf
import numpy as np
from Final_Task.network import Policy
from Final_Task.training import PPO_gym_training


''' Training Parameters '''
iterations = 100
learn_rate = 0.0003
epochs = 10
batch_size = 64
discount = 0.99
gae_lambda = 0.95


''' Training '''

env = gym.make('Ant-v1')
env.seed(0)
obs = env.observation_space
acts = env.action_space

New_Policy = Policy('new_policy', env)
Old_Policy = Policy('old_policy', env)

training = PPO_gym_training()


saver = tf.train.Saver

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./log/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    print("hi")
    variables_names = [v.name for v in tf.global_variables()]
    print(variables_names)

