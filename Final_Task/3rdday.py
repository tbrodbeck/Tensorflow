import tensorflow as tf
import numpy as np
import gym
from CustomCell import CustomBasicLSTMCell


class Training_util:
    ### TODO: maybe add a is_multiplayer_env param
    ### The goal of this class is to provide some useful utility for several
    ### different reinforcement learning environments, which we want to use
    ### with the PPO. It therefore allows for different creation of train_data
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
        self.envs_aggregator = []
        self.observation = []
        for elem in range(parallel_train_units):
            self.envs.append(gym.make(self.env_name))
            self.envs_aggregator.append({'action_list': [], 'value_list': [],
                                         'observation_list': [], 'reward_list'
                                         : []})
            self.observation.append(self.envs[elem].reset())
        self.gae_lambda = gae_lambda
        self.value_gamma = value_gamma
        ### trainmode switches between:
        ### 'runs' : using @parallel_train_units environments, create
        ### @train_runs runs, each with possibly different run length
        ### in this mode, train_samples is not used
        ### 'horizon': create train_runs runs of length
        ### @train_samples. train_runs should be multiple of
        ### parallel_train_units in this case
        assert train_mode in ['runs', 'horizon']
        self.train_mode = train_mode
        self.train_runs = train_runs
        self.train_samples = train_samples

    def create_train_data_step(self, actions, value_estimate):
        ### actions is numpy array of parallel_train_units, action_size
        ### this calls step(actions[respective_env_num]) on each env
        ### returns (is_done, observations, resets):
        ### is_done is true iff train_runs is reached
        ### resets is list of size(parallel_train_units), detailing, which
        ### train_units are reset (and should have their cellstate cleared
        ### respectively), where 1: should be reset, 0: should not be reset
        actions = np.split(actions, len(actions[:, 0]))
        value_estimate = np.split(value_estimate, len(value_estimate))
        if self.train_mode == 'runs':
            return self._create_train_data_step_runs(actions, value_estimate)
        elif self.train_mode == 'horizon':
            return self._create_train_data_step_horizon(actions, value_estimate)

    def _create_train_data_step_runs(self, actions, value_estimate):
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        self.observation.clear()
        resets = []
        for action, value, env, env_aggregator in zip(actions, value_estimate, self.envs,
                                                      self.envs_aggregator):
            new_observation, reward, run_done, _ = env.step(action)
            self.observation.append(new_observation)
            env_aggregator['action_list'].append(action)
            env_aggregator['value_list'].append(value)
            env_aggregator['observation_list'].append(new_observation)
            env_aggregator['reward_list'].append(reward)
            if run_done:
                self._add_run_to_train_data(env_aggregator)
            resets.append[run_done]
        is_done = self.train_runs <= len(self.train_data)
        self.observation = np.stack(self.observation)
        return (is_done, resets)

    def _create_train_data_step_horizon(self, actions, value_estimate):
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        resets = []
        self.observation.clear()
        for action, value, env, env_aggregator in zip(actions, value_estimate, self.envs,
                                                      self.envs_aggregator):
            new_observation, reward, run_done, _ = env.step(action)
            self.observation.append(new_observation)
            env_aggregator['action_list'].append(action)
            env_aggregator['value_list'].append(value)
            env_aggregator['observation_list'].append(new_observation)
            env_aggregator['reward_list'].append(reward)
            run_done = run_done or len(env_aggregator['reward_list']) >= self.train_samples
            self.train_samples
            if run_done:
                self._add_run_to_train_data(env_aggregator)
            resets.append[run_done]
        is_done = self.train_runs <= len(self.train_data)
        self.observation = np.stack(self.observation)
        return (is_done, resets)

    def _add_run_to_train_data(self, env_aggregator):
        """The train_data needs to know about:
            action - is used by L_clip
            advantage (gae)
            v_targ
            v_estimate
            reward
            observations
            """
        ### compute the v_targs
        rewards = np.asarray(env_aggregator['reward_list'])
        ### v_targ[t] is reward + gamma*v_targ[t+1], is reward at T(T = max_t)
        v_targs = np.copy(rewards)
        for reverse_index in [-i for i in range(2, len(v_targs))]:
            v_targs[reverse_index] += self.value_gamma * v_targs[reverse_index + 1]
        ### Implement gae - generalized advantage estimation
        ### shift values one to left and discount them by factor gamma
        shifted_discounted_value_estimations = \
            np.asarray(env_aggregator['value_list'] + [0])[1:] * self.value_gamma
        delta_t = rewards + shifted_discounted_value_estimations - \
                  np.asarray(env_aggregator['value_list'])
        gae_advantage = delta_t
        for reverse_index in [-i for i in range(2, len(delta_t))]:
            gae_advantage[reverse_index] += self.gae_lambda * self - value_gamma\
                                            * gae_advantage[reverse_index + 1]
        ### put everything into the train_data
        run = {'action': np.stack(env_aggregator['action_list']), 'advantage':
            gae_advantage, 'v_targ': v_targs, 'v_estimate': np.stack(
            env_aggregator['value_list']),
               'reward': rewards, 'observation':
                   np.stack(env_aggregator['observation_list'])}
        self.train_data.append(run)
        for aggregator_list in env_aggregator.values():
            aggregator_list.clear()

    def get_observation(self):
        return self.observation


class network:
    ### We define the network here
    def __init__(self, num_units, observation_size, name, action_size,
                 batch_size, weights=None):
        print('numunits' + str(num_units))
        with tf.variable_scope(name):
            initializer = tf.initializers.truncated_normal()
            if weights is not None:
                self.batch_size = batch_size
                self.observation = tf.placeholder(tf.float32, shape
                =(self.batch_size, observation_size))
                self.embedding_weights = tf.Variable(tf.constant(
                    weights['embedding_weights']))
                self.embedding_bias = tf.Variable(tf.constant(
                    weights['embedding_bias']))
                self.embedding = tf.nn.relu_layer(self.observation,
                                                  self.embedding_weights,
                                                  self.embedding_bias)
                self.lstm = CustomBasicLSTMCell(num_units, custom_settings=True,
                                                custom_bias=weights['lstm_bias'],
                                                custom_weights=weights[
                                                    'lstm_weights'])
                # Define readout parameters
                self.action_mu_readout_weights = tf.Variable(tf.constant(
                    weights['action_mu_readout_weights']))
                self.action_mu_readout_bias = tf.Variable(tf.constant(
                    weights['action_mu_readout_bias']))
                self.action_sigma_readout_bias = tf.Variable(tf.constant(
                    weights['action_sigma_readout_bias']))
                self.action_sigma_readout_weights = tf.Variable(tf.constant(
                    weights['action_sigma_readout_weights']))
                self.value_readout_weights = tf.Variable(tf.constant(
                    weights['value_readout_weights']))
                self.value_readout_bias = tf.Variable(tf.constant(
                    weights['value_readout_bias']))
                self.state = self.lstm.zero_state(batch_size)
            else:
                self.batch_size = batch_size
                self.observation = tf.placeholder(tf.float32,
                                                  shape=(self.batch_size,
                                                         observation_size))
                self.embedding_weights = tf.Variable(initializer((
                    observation_size, num_units)))
                self.embedding_bias = tf.Variable(initializer((1, num_units)))
                self.embedding = tf.nn.relu_layer(self.observation,
                                                  self.embedding_weights,
                                                  tf.squeeze(
                                                      self.embedding_bias))
                self.lstm = CustomBasicLSTMCell(num_units)
                # Define readout parameters
                self.action_mu_readout_weights = tf.Variable(initializer(
                    shape=(num_units, action_size)))
                self.action_mu_readout_bias = tf.Variable(initializer(
                    shape=(1, action_size)))
                self.action_sigma_readout_bias = tf.Variable(initializer(
                    shape=(1, action_size)))
                self.action_sigma_readout_weights = tf.Variable(initializer(
                    shape=(num_units, action_size)))
                self.value_readout_weights = tf.Variable(initializer(
                    shape=(num_units, 1)))
                self.value_readout_bias = tf.Variable(initializer(shape=(1, 1)))
                self.state = self.lstm.zero_state(batch_size, dtype=tf.float32)

    def step(self, name):
        with tf.variable_scope(name):

            zero_one = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
            self.newstate_c = tf.multiply(self.state.c, zero_one)
            self.newstate_h = tf.multiply(self.state.h, zero_one)

            lstm_out, self.state = self.lstm(self.embedding, self.state)
            value = tf.add(tf.matmul(lstm_out, self.value_readout_weights),
                           self.value_readout_bias)
            action_mu=tf.add(tf.matmul(lstm_out, self.action_mu_readout_weights),self.action_mu_readout_bias)
            action_sigma=tf.add(tf.matmul(lstm_out,
                                          self.action_mu_readout_weights), self.action_sigma_readout_bias)
            action = tf.distributions.Beta(action_mu, action_sigma).sample()
            return value ,action_mu, action_sigma, action


    def reset_states(self, resets):
        return this_should_change_some_states


# number of iterations for the whole algorithm
iteration_num = 10
# name of the openaigym used
env_name = 'Ant-v1'
# size of the observation
observation_size = gym.make(env_name).observation_space.shape[0]
# size of an action (the output for our policy has to be twice as big, as we
# have to model a probability density function pdf over it)
action_size = gym.make(env_name).action_space.shape[0]
#how many environments should be used to generate train_data at once
parallel_envs = 2
#batch size for network in creating training data
batch_size_data_creation = parallel_envs
# size of a minibatch in in optimization
batch_size_parameter_optimization = 100
# amount of epochs to train over one set of training_data
optimization_epochs = 100
# size of the lstm cell
lstm_unit_num = 128
# gamma value for discounting rewards for value function
value_gamma = 0.99
# lambda value for generalized advantage estimator gae values
gae_lambda = 0.99
# amount of training runs to assemble for one training-optimization iteration
train_runs = 200
# length of one training run, THIS IS NOT USED IN 'runs'
train_run_length = 400
# length of the subsequences we will train on
training_sequence_length = 100
# how many full episodes of training are performed on one trainingset
optimization_batches = 10
# def __init__(self, weights, parallel_train_units, gae_lambda, value_gamma,
#                 env_name, train_runs, train_mode, train_samples = None):
utility = Training_util(None, parallel_envs, gae_lambda, value_gamma,
                        env_name, train_runs, 'horizon', train_run_length)

for iteration in range(iteration_num):
    # deploy a new graph for every new training_iteration, minimizing the trash
    # left over in our RAM
    graph = tf.Graph()
    ### First we build the train_data_set
    # The old network generates train_samples
    train_data_network = None
    with graph.as_default():
        step = 0
        done = False
        if iteration == 0:
            train_data_network = network(lstm_unit_num, observation_size,
                                         'iteration' + str(iteration) + 'train_data_generation', action_size,
                                         batch_size_data_creation)
        else:
            train_data_network=network(num_units,
                                       observation_size,'iteration'+str(iteration)+'train_data_generation',
                                       action_size, batch_size_data_creation, utility.weights)
        while not done:
            print('not_done')
            value, mu, sigma, action = train_data_network.step('unfold_iteration'+str(iteration)+'step'+str(step))
            step+=1
            with tf.Session(graph = graph) as session:
                session.run(tf.global_variables_initializer())
                value, mu, sigma, action = session.run((value, mu, sigma,
                                                        action),
                            feed_dict={train_data_network.observation:utility.get_observation()})
                print(mu)
                print(sigma)
                print(value)
                print(action)
                print(batch_size_data_creation)

