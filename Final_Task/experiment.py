import tensorflow as tf
import numpy as np
from Final_Task.CustomCell import CustomBasicLSTMCell
import gym
import gym_compete

import tensorflow as tf
import numpy as np
import gym


class training_util:
    ### we need to save the weights, keep track of our gyms and the train_data
    ### stemming from the gyms
    ### Also a smart way to refine and provide the training samples is needed
    ###
    def __init__(self, weights, parallel_train_units, gae_lambda, value_gamma,
                 env_name, train_runs, train_mode, train_samples=None):
        self.weights = weights
        self.train_data = []
        self.parallel_train_units = parallel_train_units
        self.env_name = env_name
        self.envs = []
        selv.envs_aggregator = []
        for elem in parallel_train_units:
            self.envs.append[gym.make(self.env_name)]
            self.envs_aggregator.append([])
            self.envs[elem].reset()
        self.gae_lambda = gae_lambda
        self.value_gamma = value_gamma
        ### trainmode switches between:
        ### 'runs' : using @parallel_train_units environments, create
        ### @train_runs runs, each with possibly different run length
        ### in this mode, @train_samples is not used
        ### 'horizon': create train_runs runs of length
        ### @train_samples. train_runs should be multiple of
        ### parallel_train_units in this case
        assert train_mode in ['runs', 'horizon']
        self.train_mode = train_mode
        self.train_runs = train_runs
        self.train_samples = train_samples

    def create_train_data_step(self, actions):
        ### actions is numpy array of parallel_train_units, action_size
        ### this calls step(actions[respective_env_num]) on each env
        ### returns (is_done, observations, resets):
        ### is_done is true iff train_runs is reached
        ### observations is list of respective observations
        ### resets is list of size(parallel_train_units), detailing, which
        ### train_units are reset (and should have their cellstate cleared
        ### respectively), where 1: should be reset, 0: should not be reset
        if self.train_mode = 'runs':
            return self._create_train_data_step_runs(actions)
        elif self.train_mode= 'horizon':
            return self._create_train_data_step_horizon(actions)

    def _create_train_data_step_runs(self, actions):
        assert len(actions) == len(self.envs)
        resets = []
        observations = []
        for action, env, env_aggregator in zip(actions, self.envs,
                                               selv.envs_aggregator):
            observation, reward, run_done, _ = env.step(action)
            env_aggregator.append((action, observation, reward))
            if run_done:
                self._add_run_to_train_data(env_aggregator)
            resets.append[run_done]
            observations.append(observation)
        done = self.train_runs <= len(self.train_data)

    def _add_run_to_train_data(self, env_aggregator):
        """The train_data needs to know about:
            action - is used by L_clip
            advantage (gae)"""
        for t in env_aggregator:
                value.append(0)
                delta = reward + gamma * value[t+1] - v[t]
                gae = rew

        env_aggregator.clear()


batch_num = 4
input_shape = (5, 1, 4)
lstm_size = 4
num_units = lstm_size
input_length = 5
predefined_input = np.random.rand(input_shape[0], input_shape[1], input_shape[2])
print(predefined_input)
lstm_params = None
for batch in range(batch_num):
    graph = tf.Graph()
    with graph.as_default():
        input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
        print('creating')
        print(batch)
        if batch == 0:
            lstm = CustomBasicLSTMCell(lstm_size)
        else:
            lstm = CustomBasicLSTMCell(lstm_size, custom_settings=True,
                                       custom_bias=tf.initializers.constant(lstm_params[1]),
                                       custom_weights=tf.initializers.constant(lstm_params[0]))
        # print('building')
        lstm_results = []
        # print('setting state')
        state = lstm.zero_state(1, tf.float32)
        # print('unfolding')
        for t_step in range(input_length):
            output, state = lstm(input_placeholder[t_step, :, :], state)
            lstm_results.append(output)
        results = tf.stack(lstm_results)
        result_mean = tf.reduce_min(results)
        train_step = tf.train.AdamOptimizer().minimize(result_mean)
        # print('session')
        with tf.Session(graph=graph) as session:
            # print('initializing')
            session.run(tf.global_variables_initializer())
            # print('running')
            outcome_mean = session.run((result_mean), feed_dict={input_placeholder:
                                                                     predefined_input})
            print(outcome_mean)
            lstm_params = session.run(lstm.parameters)
        # print(lstm_params)
