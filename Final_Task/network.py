import gym
import numpy as np
import tensorflow as tf


class Policy:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        ob_space = env.observation_space
        act_space = env.action_space


        #with tf.variable_scope(name):
        #    pass