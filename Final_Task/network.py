import gym
import numpy as np
import tensorflow as tf
from Final_Task.CustomCell import CustomBasicLSTMCell


class Policy:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        ob_space = env.observation_space
        act_space = env.action_space

        self.lstm_memory_size = ob_space.shape[0]
        self.subsequence_length = 3
        self.batch_size = 1
        self.parameters = []


        with tf.variable_scope(name):
            obs = tf.placeholder(dtype=tf.float32,
                                      shape=[self.batch_size,
                                             self.subsequence_length,
                                             self.lstm_memory_size],
                                      name='obs')
            sequences = tf.unstack(obs, self.subsequence_length, axis=1)

            cell_state = tf.placeholder(tf.float32,
                                        shape=[self.batch_size,
                                               self.lstm_memory_size])
            hidden_state = tf.placeholder(tf.float32,
                                          shape=[self.batch_size,
                                                 self.lstm_memory_size])
            cell = CustomBasicLSTMCell(self.lstm_memory_size)


            zero_state = cell.zero_state(self.batch_size, tf.float32)
            state = tf.nn.rnn_cell.LSTMStateTuple(c=cell_state, h=hidden_state)

            outputs, state = tf.nn.static_rnn(cell, sequences,
                                              initial_state=state)
            outputs = tf.reshape(tf.concat(outputs, 1), [self.batch_size,
                                                         self.subsequence_length,
                                                         self.lstm_memory_size])

            self.parameters.append(cell.parameters)

    @property
    def parameters(self):
        return self.parameters

'''
# Test
env = gym.make('Ant-v1')
p = Policy('p', env)
'''
'''
with tf.Session as s:
    out_1, state_1 = s.run((p.obs),
                                 feed_dict={obs: n})
'''
