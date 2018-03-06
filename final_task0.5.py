import tensorflow as tf
import numpy as np
import gym
from CustomCell import CustomBasicLSTMCell
import random


class Training_util:
    ### TODO: maybe add a is_multiplayer_env param

    def __init__(self, weights, parallel_train_units, gae_lambda, value_gamma,
                 env_name, train_runs, train_mode, horizon=None):
        """
        The goal of this class is to provide some useful utility for several
        different reinforcement learning environments, which we want to use
        with the PPO. It therefore allows for different creation of train_data
        we need to save the weights, keep track of our gyms and the train_data
        stemming from the gyms
        Also a smart way to refine and provide the training samples is needed

        :param weights: weights of trainable policy
        :param parallel_train_units: number of parallel trained gym
        environments
        :param gae_lambda: parameter of objective function
        :param value_gamma: parameter of objective function
        :param env_name: name of @gym.env
        :param train_runs: number of runs
        :param train_mode: 'runs'- or 'horizon'-mode
        :param horizon: the amount of samples in case of 'horizon'
        """

        self.weights = weights
        self.train_data = []
        self.parallel_train_units = parallel_train_units
        self.env_name = env_name
        self.envs = []
        self.envs_aggregator = []
        self.observation = []

        print('Creating parallel environments')

        for elem in range(parallel_train_units):
            self.envs.append(gym.make(self.env_name))
            self.envs_aggregator.append({'action_list': [], 'value_list': [],
                                         'observation_list': [], 'reward_list'
                                         : [], 'alpha_list': [], 'beta_list':[]})
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
        self.horizon = horizon

    def create_train_data_step(self, actions, value_estimate, alpha, beta):
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
            return self._create_train_data_step_runs(actions, value_estimate,
                                                     alpha, beta)
        elif self.train_mode == 'horizon':
            return self._create_train_data_step_horizon(actions,
                                                        value_estimate, alpha,
                                                       beta)

    def _create_train_data_step_runs(self, actions, value_estimate, alpha, beta):
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        self.observation = []
        resets = []
        for action, value, env, env_aggregator, alpha_val, beta_val in zip(actions, value_estimate, self.envs,
                                                      self.envs_aggregator,
                                                                           alpha,
                                                                          beta):
            new_observation, reward, run_done, _ = env.step(action)
            self.observation.append(new_observation)
            env_aggregator['alpha_list'].append(alpha)
            env_aggregator['beta_list'].append(beta)
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

    def _create_train_data_step_horizon(self, actions, value_estimate, alpha,
                                        beta):
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        resets = []
        self.observation = []
        for action, value, env, env_aggregator, alpha_val, beta_val in zip(actions, value_estimate, self.envs,
                                                      self.envs_aggregator,
                                                                           alpha,
                                                                          beta):
            new_observation, reward, run_done, _ = env.step(action)
            self.observation.append(new_observation)
            env_aggregator['alpha_list'].append(alpha_val)
            env_aggregator['beta_list'].append(beta_val)
            env_aggregator['action_list'].append(action)
            env_aggregator['value_list'].append(value)
            env_aggregator['observation_list'].append(new_observation)
            env_aggregator['reward_list'].append(reward)
            run_done = run_done or len(env_aggregator['reward_list']) >= self.horizon
            self.horizon
            if run_done:
                self._add_run_to_train_data(env_aggregator)
            resets.append(run_done)
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
            gae_advantage[reverse_index] += self.gae_lambda * self.value_gamma * gae_advantage[reverse_index + 1]
        ### put everything into the train_data
        run = {'action': np.stack(env_aggregator['action_list']), 'advantage':
            gae_advantage, 'v_targ': v_targs, 'v_estimate': np.stack(
            env_aggregator['value_list']),
               'reward': rewards, 'observation':
               np.stack(env_aggregator['observation_list']), 'alpha':
               np.stack(env_aggregator['alpha_list']),
               'beta':np.stack(env_aggregator['beta_list'])}
        print('appending_now')
        self.train_data.append(run)
        for aggregator_list in env_aggregator.values():
            aggregator_list.clear()

    def get_observation(self):
        return_value = self.observation

        return self.observation

    def index_train_samples(self, length):
        """
        Returns a list of dictionaries with sequences of the length @length
        which is shuffled.
        Indexes the correspondung run, the beginnung and the end of a
        sequence.

        :param length: int
        :return: list of dictionaries
        """
        dicts = []
        assert length <= self.horizon

        for run_nr, run in enumerate(train_data):

            # chose on training occurance
            training = run['action']

            # amount of samples we have to skip
            cutout = len(training) % length

            # by chance choose the front-part of the trajectory
            if bool(random.getrandbits(1)):
                trajectory = training[:(len(training)) - cutout]
                amount = len(trajectory)//length
                for index in range(amount):
                    beginn = index * length
                    ending = index * length + length

            # or choose the back-part of the trajectory
            else:
                trajectory = training[cutout - 1:]
                amount = len(trajectory) // length
                for index in range(amount):
                    beginn = index * length + cutout
                    ending = index * length + length + cutout

            dicts.append({'run': run_nr, 'beg': beginn, 'end': ending})

        random.shuffle(dicts)
        return dicts


class Network:
    def __init__(self, num_units, observation_size, name, action_size,
                 batch_size, weights=None):
        """
        We define the network here.

        :param num_units: int, lstm_unit_num
        :param observation_size: int, size of respective env.observation_space
        :param name: str
        :param action_size: int, size of respective env.action_space
        :param batch_size: int
        :param weights: parameters of policy
        """
        print('Network with numunits' + str(num_units))

        with tf.variable_scope(name):
            initializer = tf.initializers.truncated_normal()

            # if weights are set
            if weights is not None:
                self.batch_size = batch_size
                self.action_size = action_size 
                self.observation_size = observation_size 
                self.observation = tf.placeholder(tf.float32, shape
                =(self.batch_size, observation_size))
                self.embedding_weights = tf.Variable(tf.constant(
                    weights['embedding_weights']))
                self.embedding_bias = tf.Variable(tf.constant(
                    weights['embedding_bias']))
                self.embedding = tf.nn.relu_layer(self.observation,
                                                  self.embedding_weights,
                                                  tf.squeeze(self.embedding_bias))
                self.lstm = CustomBasicLSTMCell(num_units, custom_settings=True,
                                                custom_bias=weights['lstm_bias'],
                                                custom_weights=weights[
                                                    'lstm_weights'])
                # Define readout parameters
                self.action_alpha_readout_weights = tf.Variable(tf.constant(
                    weights['action_alpha_readout_weights']))
                self.action_alpha_readout_bias = tf.Variable(tf.constant(
                    weights['action_alpha_readout_bias']))
                self.action_beta_readout_bias = tf.Variable(tf.constant(
                    weights['action_beta_readout_bias']))
                self.action_beta_readout_weights = tf.Variable(tf.constant(
                    weights['action_beta_readout_weights']))
                self.value_readout_weights = tf.Variable(tf.constant(
                    weights['value_readout_weights']))
                self.value_readout_bias = tf.Variable(tf.constant(
                    weights['value_readout_bias']))
                self.state = self.lstm.zero_state(batch_size, dtype=tf.float32)
            else:
                self.batch_size = batch_size
                self.action_size = action_size 
                self.observation_size = observation_size 
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
                self.action_alpha_readout_weights = tf.Variable(initializer(
                    shape=(num_units, action_size)))
                self.action_alpha_readout_bias = tf.Variable(initializer(
                    shape=(1, action_size)))
                self.action_beta_readout_bias = tf.Variable(initializer(
                    shape=(1, action_size)))
                self.action_beta_readout_weights = tf.Variable(initializer(
                    shape=(num_units, action_size)))
                self.value_readout_weights = tf.Variable(initializer(
                    shape=(num_units, 1)))
                self.value_readout_bias = tf.Variable(initializer(shape=(1, 1)))
                self.state = self.lstm.zero_state(batch_size, dtype=tf.float32)

    def step(self, name, step, truncation_factor):
        """
        we run the policy for one step and retrieve the observable data

        :param name: str, step name
        :param step: int
        :param truncation_factor: int
        :return: value, action_alpha, action_beta, action
        """
        with tf.variable_scope(name):
            #zero_one = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
            #self.newstate_c = tf.multiply(self.state.c, zero_one)
            #self.newstate_h = tf.multiply(self.state.h, zero_one)
            #TODO maybe simply saving weights in a variable would be an option
            if step%truncation_factor == 0:
                self.state = self.lstm.zero_state(self.batch_size, dtype =
                                                  tf.float32)
            lstm_out, self.state = self.lstm(self.embedding, self.state)
            value = tf.add(tf.matmul(lstm_out, self.value_readout_weights),
                           self.value_readout_bias)
            action_alpha=tf.nn.relu( tf.add(tf.matmul(lstm_out,
                                                      self.action_alpha_readout_weights),self.action_alpha_readout_bias))
            action_beta=tf.nn.relu(tf.add(tf.matmul(lstm_out,
                                          self.action_beta_readout_weights),
                                           self.action_beta_readout_bias))
            action = tf.distributions.Beta(action_alpha, action_beta).sample()
            return value, action_alpha, action_beta, action

    def optimize(self, name, sequence_length, learn_rate, epsilon, c1, c2):
        """
        feed chunks of length sequence_length
        optimize weights based on: 
            alpha, beta
            advantage
            observation
            target_value
            action 
        """
        with tf.variable_scope(name):
            
            self.alpha = tf.placeholder(dtype=tf.float32,
                                        shape=(sequence_length,self.batch_size,self.action_size))
            self.beta = tf.placeholder(dtype=tf.float32,
                                        shape=(sequence_length,self.batch_size,self.action_size))
            self.action = tf.placeholder(dtype=tf.float32,
                                        shape=(sequence_length,self.batch_size,self.action_size))
            self.gae_advantage = tf.placeholder(dtype=tf.float32,
                                        shape=(sequence_length,self.batch_size,1))
            self.target_value = tf.placeholder(dtype=tf.float32,
                                        shape=(sequence_length,self.batch_size,1))
            self.optimization_observation = tf.placeholder(dtype = tf.float32,
                                                           shape =
                                                           (sequence_length,
                                                            self.batch_size,
                                                            self.observation_size))
            self.state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
            for t_step in range(sequence_length):
                embedding = tf.nn.relu_layer(tf.squeeze(self.optimization_observation[t_step,:,:]), self.embedding_weights, tf.squeeze(self.embedding_bias))
                lstm_out, self.state = self.lstm(embedding, self.state)
                val_pred = tf.add(tf.matmul(embedding, self.value_readout_weights),
                                 tf.squeeze(self.value_readout_bias))
                alpha_pred = tf.nn.relu_layer(embedding,
                                              self.action_alpha_readout_weights,
                                              tf.squeeze(self.action_alpha_readout_bias))
                beta_pred = tf.nn.relu_layer(embedding,
                                              self.action_beta_readout_weights,
                                              tf.squeeze(self.action_beta_readout_bias))
                policy_new = tf.distributions.Beta(alpha_pred, beta_pred)
                policy_old = tf.distributions.Beta(self.alpha, self.beta)
                prob_new = policy_new.prob(tf.squeeze(self.action[t_step,:,:]))
                prob_old = policy_old.prob(tf.squeeze(self.action[t_step,:,:]))
                entropy = policy_new.entropy()
                entropy_product = tf.ones(self.batch_size)
                prob_product_old = tf.ones(self.batch_size)
                prob_product_new = tf.ones(self.batch_size)
                for i in range(self.action_size):
                    prob_product_old = tf.multiply(prob_product_old, prob_old[:,i])
                    prob_product_new = tf.multiply(prob_product_new,
                                                   prob_new[:,i])
                    entropy_product = tf.multiply(entropy_product, entropy[:,i])

                L_explore = entropy_product
                L_value = tf.losses.mean_squared_error(val_pred,
                                                       self.target_value[t_step,
                                                                          :,:])
                prob_ratio = tf.divide(prob_product_new, prob_product_old)
                L_clip = tf.minimum(self.gae_advantage[t_step, :,:]*prob_ratio,
                                   tf.clip_by_value(prob_ratio, 1-epsilon,
                                                    1+epsilon)*self.gae_advantage[t_step,:,:])
                return L_explore, L_value, L_clip 

        
    def reset_states(self, resets):
        return this_should_change_some_states

    def network_parameters(self):
        kernel, bias = self.lstm.parameters
        embedding_weights = self.embedding_weights
        embedding_bias = self.embedding_bias
        value_readout_weights = self.value_readout_weights
        value_readout_bias = self.value_readout_bias
        action_alpha_readout_weights = self.action_alpha_readout_weights
        action_alpha_readout_bias = self.action_alpha_readout_bias
        action_beta_readout_weights = self.action_beta_readout_weights
        action_beta_readout_bias = self.action_beta_readout_bias
        return kernel, bias, embedding_weights, embedding_bias, value_readout_weights, value_readout_bias, action_alpha_readout_weights, action_alpha_readout_bias, action_beta_readout_weights, action_beta_readout_bias 

# keys is needed to build weight-dictonary for weight memory
keys = ['lstm_weights', 'lstm_bias', 'embedding_weights',
        'embedding_bias', 'value_readout_weights',
        'value_readout_bias', 'action_alpha_readout_weights',
        'action_alpha_readout_bias',
        'action_beta_readout_weights', 'action_beta_readout_bias']


''' Hyperparameters '''

# number of iterations for the whole algorithm
iteration_num = 1

# name of the openaigym used
env_name = 'Ant-v1'

# size of the observation
observation_size = gym.make(env_name).observation_space.shape[0]

# size of an action (the output for our policy has to be twice as big, as we
# have to model a probability density function pdf over it)
action_size = gym.make(env_name).action_space.shape[0]
#how many environments should be used to generate train_data at once
parallel_envs = 100
#batch size for network in creating training data
batch_size_data_creation = parallel_envs

# size of a minibatch in in optimization
batch_size_parameter_optimization = 1
# amount of epochs to train over one set of training_data
optimization_epochs = 100

# size of the lstm cell
lstm_unit_num = 128

# gamma value for discounting rewards for value function
value_gamma = 0.99

# lambda value for generalized advantage estimator gae values
gae_lambda = 0.99

# amount of training runs to assemble for one training-optimization iteration
train_runs = 100
# length of one training run, THIS IS NOT USED IN 'runs'
horizon = 30
# length of the subsequences we will train on
training_sequence_length = horizon
assert training_sequence_length <= horizon
# how many full episodes of training are performed on one trainingset
optimization_batches = 10

# train mode, either 'horizon' or 'runs'
train_mode = 'horizon'

#truncation factor USE IN 'horizon' MODE ONLY!
truncation_factor = 50
#learn_rate
learn_rate = 0.01
#epsilon
epsilon = 0.2
#c1, hyperparameter factor for weighting l_value loss
c1 = 1
#c2, hyperparameter factor for weighting l_exploration loss
c2 = 1
### This utility class saves weights and keeps track of the training_data
utility = Training_util(None, parallel_envs, gae_lambda, value_gamma,
                        env_name, train_runs, train_mode, horizon)

print('Start training!')

for iteration in range(iteration_num):
    # deploy a new graph for every new training_iteration, minimizing the
    # trash left over in our RAM
    graph = tf.Graph()

    # First we build the train_data_set
    # The old network generates train_samples
    train_data_network = None

    with graph.as_default():
        step = 0
        done = False

        # create or retrieve corresponding network for new iteration
        if iteration == 0:
            train_data_network = Network(lstm_unit_num, observation_size,
                          'iteration'+str(iteration)+'train_data_generation',
                          action_size, batch_size_data_creation)

        else:
            train_data_network = Network(lstm_unit_num, observation_size, 'iteration'+str(iteration)+'train_data_generation', action_size, batch_size_data_creation, utility.weights)

        while not done:
            # generate one step of the policy
            training, alpha, beta, action = train_data_network.step('unfold_iteration' + str(iteration) + 'step' + str(step),
                                                                    step, truncation_factor)
            with tf.Session(graph = graph) as session:
                session.run(tf.global_variables_initializer())
                training, alpha, beta, action = session.run((training, alpha, beta,
                                                             action),
                                                            feed_dict={train_data_network.observation:utility.get_observation()})
                is_done, resets = utility.create_train_data_step(action, training, alpha, beta)
                print('iteration: '+str(iteration))
                print('step: '+str(step))

                step+=1
                done = is_done

        print('next iteration')
        print(len(utility.train_data))
        print(utility.train_data[0]['alpha'])
        if iteration == 0: 
            with tf.Session(graph = graph) as session:
                session.run(tf.global_variables_initializer())
                parameters = session.run(train_data_network.network_parameters())
                weights = {}

                #keys are defined in the hyperparameter list
                for parameter, key in zip(parameters, keys):
                    weights[key] = parameter
                train_data = utility.train_data 
                utility = Training_util(weights, parallel_envs, gae_lambda, value_gamma, env_name, train_runs, train_mode, horizon)
                utility.train_data = train_data


    # Now we got the trian_data
    print("There we are")
    print(len(utility.train_data))
    print(utility.index_train_samples(7))

    graph = tf.Graph()
    with graph.as_default():
        optimizing_network=Network(lstm_unit_num,
                                   observation_size,'iteration'+str(iteration)+'optimization',
                                   action_size,
                                   batch_size_parameter_optimization, utility.weights)

        ### and now we have to implement the training procedure
        for epoch in range(optimization_epochs):
            #this is messy, might still work
            used_samples = None
            train_sample_plan = np.arange(len(utility.train_data))
            np.random.shuffle(train_sample_plan)
            train_data = utility.train_data

            # let's try SGD first
            for indices in train_sample_plan.tolist():
                alpha = np.stack([train_data[i]['alpha'] for i in index], axis = 1)
                beta = np.stack([train_data[i]['beta'] for i in index], axis = 1)
                advantages = np.stack([train_data[i]['gae_advantage'] for i in index], axis = 1)
                v_targ = np.stack([train_data[i]['v_targ'] for i in index], axis = 1)
                action = np.stack([train_data[i]['action'] for i in index], axis = 1)
                observation = np.stack([train_data[i]['observation'] for i in index], axis = 1)
                l_clip, l_v, l_e = None
                optimizing_network.optimize('iteration'+str(iteration)+'optimizationepoch'+
                                            str(epoch),training_sequence_length,
                                                            learn_rate,
                                                            epsilon, c1, c2),
                with tf.Session(graph = graph) as session:
                    print(utility.train_data[i]['alpha'].shape)
                    #can not preconstruct initializer, as new variables are added
                    session.run(tf.global_variables_initializer())
                    session.run(l_clip, l_v, l_e, feed_dict =
                                {optimizing_network.alpha: alpha, 
                                 optimizing_network.beta: beta, 
                                 optimizing_network.gae_advantage:advantages,
                                 optimizing_network.target_value:v_targ, 
                                 optimizing_network.action:action, 
                                 optimizing_network.optimization_observation:observation})
        with tf.Session(graph = graph) as session:
            session.run(tf.global_variables_initializer())
            parameters = session.run(train_data_network.network_parameters())
            weights = {}

            #keys are defined in the hyperparameter list
            for parameter, key in zip(parameters, keys):
                weights[key] = parameter 
            utility = Training_util(weights, parallel_envs, gae_lambda, value_gamma,
                                    env_name, train_runs, train_mode,
                                    horizon)

            
                
