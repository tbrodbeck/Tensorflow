import tensorflow as tf
import numpy as np
import gym
from custom_lstm_cell import CustomBasicLSTMCell

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
        self.train_samples = train_samples

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
            env_aggregator['value_list'].append(np.squeeze(value))
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
            env_aggregator['action_list'].append(np.squeeze(action))
            env_aggregator['value_list'].append(np.squeeze(value))
            env_aggregator['observation_list'].append(new_observation)
            env_aggregator['reward_list'].append(reward)
            run_done = run_done or len(env_aggregator['reward_list']) >= self.train_samples
            self.train_samples
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
        #TODO: Can we go through both reverse indices at once?
        v_targs = np.copy(rewards)
        for reverse_index in [-i for i in range(2, len(v_targs))]:
            v_targs[reverse_index] += self.value_gamma * v_targs[reverse_index + 1]
        ### Implement gae - generalized advantage estimation
        ### shift values one to left and discount them by factor gamma
        shifted_discounted_value_estimations = np.asarray(env_aggregator['value_list'] + [0])[1:] * self.value_gamma
        #print('shifted_discounted_value_estimations_shape'
        #      +str(shifted_discounted_value_estimations.shape))
        #print('rewards_shape' + str(rewards.shape))
        #print('value_shapes' +
        #      str(np.asarray(env_aggregator['value_list']).shape))
        delta_t = rewards + shifted_discounted_value_estimations - np.asarray(env_aggregator['value_list'])
        #print('shape of delta_t' + str(delta_t.shape))
        gae_advantage = delta_t
        for reverse_index in [-i for i in range(2, len(delta_t))]:
            gae_advantage[reverse_index] += self.gae_lambda * self.value_gamma * gae_advantage[reverse_index + 1]
        ### put everything into the train_data
        #print('advantage-shape:'+ str(gae_advantage.shape))
        run = {'action': np.stack(env_aggregator['action_list']), 'advantage':
            gae_advantage, 'v_targ': v_targs, 'v_estimate': np.stack(
            env_aggregator['value_list']),
               'reward': rewards, 'observation':
               np.stack(env_aggregator['observation_list']), 'alpha':
               np.stack(env_aggregator['alpha_list']),
               'beta':np.stack(env_aggregator['beta_list'])}
        self.train_data.append(run)
        for aggregator_list in env_aggregator.values():
            aggregator_list.clear()

    def get_observation(self):
        return_value = self.observation

        return self.observation


class network:
    ### We define the network here
    def __init__(self, num_units, observation_size, name, action_size,
                 batch_size, weights=None, learn_rate = None):
        with tf.variable_scope(name):
            if learn_rate is not None:
                self.optimizer = tf.train.AdamOptimizer(learn_rate)
            initializer = tf.initializers.truncated_normal()
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
    def reset_states(self):
        self.state = self.lstm.zero_state(self.batch_size, dtype = tf.float32)

    def step(self, name, step, truncation_factor):
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
            return value ,action_alpha, action_beta, action  

    def optimize(self, name, sequence_length, epsilon, c1, c2):
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
                                        shape=(sequence_length,self.batch_size))
            self.target_value = tf.placeholder(dtype=tf.float32,
                                        shape=(sequence_length,self.batch_size))
            self.optimization_observation = tf.placeholder(dtype = tf.float32,
                                                           shape =
                                                           (sequence_length,
                                                            self.batch_size,
                                                            self.observation_size))
            loss_explore = []
            loss_value = []
            loss_clip = []
            self.state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
            for t_step in range(sequence_length):
                embedding = tf.nn.relu_layer(self.optimization_observation[t_step,:,:], self.embedding_weights, tf.squeeze(self.embedding_bias))
                lstm_out, self.state = self.lstm(embedding, self.state)
                val_pred = tf.add(tf.matmul(lstm_out, self.value_readout_weights),
                                 tf.squeeze(self.value_readout_bias))
                alpha_pred = tf.nn.relu_layer(lstm_out,
                                              self.action_alpha_readout_weights,
                                              tf.squeeze(self.action_alpha_readout_bias))
                beta_pred = tf.nn.relu_layer(lstm_out,
                                              self.action_beta_readout_weights,
                                              tf.squeeze(self.action_beta_readout_bias))
                #print('alpha_pred shape')
                #print(alpha_pred.get_shape())
                #print('beta_pred_shape')
                #print(beta_pred.get_shape())
                #print('action is of size:')
                #print(self.action[0,:,:].get_shape())
                #print('squeezed action is of shape:')
                #print(self.action[0,:,:].get_shape())
                policy_new = tf.distributions.Beta(alpha_pred, beta_pred)
                policy_old = tf.distributions.Beta(self.alpha[t_step,:,:],
                                                   self.beta[t_step,:,:])
                #print('squeezed action is of size')
                #print(tf.squeeze(self.action[t_step,:,:]).get_shape())
                prob_new = policy_new.prob(tf.squeeze(self.action[t_step,:,:]))
                prob_old = policy_old.prob(tf.squeeze(self.action[t_step,:,:]))
                entropy = policy_new.entropy()
                #print('probs-new are of shape:')
                #print(prob_new.get_shape())
                #print('probs-old are of shape: ')
                #print(prob_old.get_shape())
                entropy_product = tf.ones(self.batch_size)
                prob_product_old = tf.ones(self.batch_size)
                prob_product_new = tf.ones(self.batch_size)
                for i in range(self.action_size):
                    #print('prob_prod_old_shape')
                    #print(prob_product_old.get_shape())
                    #print('prob_old_shape')
                    #print(prob_old.get_shape())
                    prob_product_old = tf.multiply(prob_product_old, prob_old[:,i])
                    prob_product_new = tf.multiply(prob_product_new, prob_new[:,i])
                    entropy_product = tf.multiply(entropy_product, entropy[:,i])
                
                l_explore = tf.reshape(entropy_product,shape=(2,1))
                #print('shape of entropy_loss' + str(l_explore.get_shape))
                #print('shape of val_pred: ' + str(val_pred.get_shape()))
                #print('shape of target_value' +
                #      str(self.target_value[t_step,:].get_shape()))
                l_value = tf.square(tf.subtract(val_pred, tf.reshape(self.target_value[t_step,:], (self.batch_size,1))))
                prob_ratio = tf.divide(prob_product_new, prob_product_old)
                l_clip = tf.minimum(self.gae_advantage[t_step,:]*prob_ratio, tf.clip_by_value(prob_ratio, 1-epsilon, 1+epsilon)*self.gae_advantage[t_step,:])
                loss_clip.append(tf.squeeze(l_clip))
                loss_value.append(tf.squeeze(l_value))
                loss_explore.append(tf.squeeze(l_explore))
            loss_clip = tf.stack(loss_clip)
            loss_value = tf.stack(loss_value)
            loss_explore = tf.stack(loss_explore)
            loss_complete = tf.add(tf.subtract(loss_clip,
                                               tf.multiply(tf.constant(c1,dtype=tf.float32),
                                                           loss_value)),
                                   tf.multiply(tf.constant(c2,dtype=tf.float32),loss_explore))
            print('shape of loss' + str(loss_complete.get_shape))
            inverted_loss = tf.multiply(tf.constant(-1, dtype = tf.float32),
                                        loss_complete)
            learn_step = self.optimizer.minimize(tf.reduce_mean(inverted_loss))
            self.state = self.lstm.zero_state(self.batch_size, dtype = tf.float32)
            return learn_step 

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


keys = ['lstm_weights', 'lstm_bias', 'embedding_weights',
        'embedding_bias', 'value_readout_weights',
        'value_readout_bias', 'action_alpha_readout_weights',
        'action_alpha_readout_bias',
        'action_beta_readout_weights', 'action_beta_readout_bias']

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
batch_size_parameter_optimization = 2
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
train_run_length = 30
# length of the subsequences we will train on
training_sequence_length = train_run_length 
assert training_sequence_length <= train_run_length 
# how many full episodes of training are performed on one trainingset
optimization_batches = 10
#train mode, either 'horizon' or 'runs'
train_mode = 'horizon'
#truncation factor USE IN 'horizon' MODE ONLY!
truncation_factor = 5
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
                        env_name, train_runs, train_mode, train_run_length)
for iteration in range(iteration_num):
    #deploy a new graph for every new training_iteration, minimizing the trash
    #left over in our RAM
    graph = tf.Graph()
    ### First we build the train_data_set
    # The old network generates train_samples
    train_data_network = None
    with graph.as_default():
        step = 0
        done = False
        if iteration == 0:
            train_data_network=network(lstm_unit_num, observation_size,'iteration'+str(iteration)+'train_data_generation', action_size, batch_size_data_creation)
        else:
            train_data_network=network(lstm_unit_num, observation_size,'iteration'+str(iteration)+'train_data_generation', action_size, batch_size_data_creation, utility.weights)
        while not done:
            value, alpha, beta, action = train_data_network.step('unfold_iteration'+str(iteration)+'step'+str(step),
                                   step, truncation_factor)
            with tf.Session(graph = graph) as session:
                session.run(tf.global_variables_initializer())
                value, alpha, beta, action = session.run((value, alpha, beta, action),
                            feed_dict={train_data_network.observation:utility.get_observation()})
                is_done, resets = utility.create_train_data_step(action, value, alpha, beta)
                print('iteration: '+str(iteration))
                print('step: '+str(step))
                if train_mode is 'horizon' and train_run_length%step==0:
                    print(resetting)
                step+=1
                done = is_done
        print('next iteration')
        if iteration == 0: 
            with tf.Session(graph = graph) as session:
                session.run(tf.global_variables_initializer())
                parameters = session.run(train_data_network.network_parameters())
                weights = {}
                #keys are defined in the hyperparameter list
                for parameter, key in zip(parameters, keys):
                    weights[key] = parameter 
                train_data = utility.train_data 
                utility = Training_util(weights, parallel_envs, gae_lambda, value_gamma, env_name, train_runs, train_mode, train_run_length)
                utility.train_data = train_data

    
    #Now we got the trian_data
    graph = tf.Graph()
    with graph.as_default():
        optimizing_network=network(lstm_unit_num,
                                   observation_size,'iteration'+str(iteration)+'optimization',
                                   action_size,
                                   batch_size_parameter_optimization,
                                   utility.weights, learn_rate = learn_rate)
        ### and now we have to implement the training procedure
        for epoch in range(optimization_epochs):
            #this is messy, might still work
            used_samples = train_runs - (train_runs%batch_size_parameter_optimization)
            train_sample_plan = np.reshape(np.arange(used_samples),
                                           (int(used_samples/batch_size_parameter_optimization),
                                           batch_size_parameter_optimization))
            np.random.shuffle(train_sample_plan)
            train_sample_plan = train_sample_plan.tolist()
            train_data = utility.train_data
            # every index actually is a list of indices

            for enum, index in enumerate(train_sample_plan):
                print('Optimization:Iteration:'+str(iteration)+'Epoch'+str(epoch)+'Run'+str(enum))
                #print(index)
                #print([train_data[i]['alpha'].shape for i in index])
                alpha = np.stack([train_data[i]['alpha'] for i in index], axis = 1)
                beta = np.stack([train_data[i]['beta'] for i in index], axis = 1)
                #print('shape_beta:' + str(beta.shape))
                advantages = np.stack([train_data[i]['advantage'] for i in index], axis = 1)
                #print('shape advantages' + str(advantages.shape))
                v_targ = np.stack([train_data[i]['v_targ'] for i in index], axis = 1)
                #print('vshape' + str(v_targ.shape))
                action = np.stack([train_data[i]['action'] for i in index], axis = 1)
                #print('action_shape: ' + str(action.shape))
                observation = np.stack([train_data[i]['observation'] for i in index], axis = 1)
                #print('observation_shape' + str(observation.shape))
                loss = optimizing_network.optimize('iteration'+str(iteration)+'optimizationepoch'+
                                            str(epoch),training_sequence_length, epsilon, c1, c2)
                with tf.Session(graph = graph) as session:
                    #can not preconstruct initializer, as new variables are added
                    session.run(tf.global_variables_initializer())
                    #print(tf.trainable_variables())
                    #print('trainable variables:')
                    session.run(loss, feed_dict = 
                                {optimizing_network.alpha: alpha, 
                                 optimizing_network.beta: beta, 
                                 optimizing_network.gae_advantage: advantages,
                                 optimizing_network.target_value: v_targ, 
                                 optimizing_network.action: action, 
                                 optimizing_network.optimization_observation: observation})
        with tf.Session(graph = graph) as session:
            session.run(tf.global_variables_initializer())
            parameters = session.run(train_data_network.network_parameters())
            weights = {}
            #keys are defined in the hyperparameter list
            for parameter, key in zip(parameters, keys):
                weights[key] = parameter 
            utility = Training_util(weights, parallel_envs, gae_lambda, value_gamma,
                                    env_name, train_runs, train_mode,
                                    train_run_length)

            
                
