import gym
import tensorflow as tf
import numpy as np
from Final_Task.network import Policy


''' Training Parameters '''
iterations = 100
horizon = 2048
learn_rate = 0.0003
epochs = 10
batch_size = 64
discount = 0.99
gae_lambda = 0.95



def main():
    env = gym.make('Ant-v1')
    env.seed(0)
    obs = env.observation_space
    acts = env.action_space
    print(acts)

    New_Policy = Policy('new policy', env)
    Old_Policy = Policy('old policy', env)



    saver = tf.train.Saver

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        print(obs)
        obs = env.reset()
        print(obs)


''' Test '''
main()



def read_training_parameters():

        for t in aggregator:
                value.append(0)
                delta = reward + gamma * value[t+1] - v[t]
                gae = rew
