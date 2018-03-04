import tensorflow as tf
import numpy as np
import gym

class PPO_Training:

    def __init__(self, env_name, parallel_train_units, train_runs, gae_lambda,
                 value_gamma, horizon=None):

        self.train_data = []
        self.parallel_train_units = parallel_train_units
        self.env_name = env_name
        self.envs = []

        self.envs_aggregator = []
        for elem in parallel_train_units:
            self.envs.append[gym.make(self.env_name)]
            self.envs_aggregator.append({'action_list': [], 'value_list' : [],
                                         'observation_list' : [], 'reward_list'
                                         : []})
            self.envs[elem].reset()

        self.gae_lambda = gae_lambda
        self.value_gamma = value_gamma

        assert train_mode in ['runs', 'horizon']
        self.train_mode = train_mode
        self.train_runs = train_runs
        self.train_samples = horizon

    def create_train_data_step(self, actions, value_estimate):
        """
        this calls step(actions[respective_env_num]) on each env

        :param actions: numpy array of parallel_train_units, action_size
        :return: (is_done, observations, resets):
            is_done: is true iff train_runs is reached
            observations: is list of respective observations
            resets: is list of size(parallel_train_units), detailing, which
                train_units are reset (and should have their cellstate cleared
                respectively), where 1: should be reset, 0: should not be reset
        """
        actions = np.split(actions, len(actions[:,0]))
        value_estimate = np.split(value_estimate, len(value_estimate))
        if self.train_mode == 'runs':
            return self._create_train_data_step_runs(actions, value_estimate)
        elif self.train_mode== 'horizon':
            return self._create_train_data_step_horizon(actions, value_estimate)

    def _create_train_data_step_runs(self, actions, value_estimate):
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        resets = []
        observations = []
        for action, value, env, env_aggregator in zip(actions, value_estimate, self.envs,
                                               self.envs_aggregator):
            observation, reward, run_done, _ = env.step(action)
            env_aggregator['action_list'].append(action)
            env_aggregator['value_list'].append(value)
            env_aggregator['observation_list'].append(observation)
            env_aggregator['reward_list'].append(reward)
            if run_done:
                self._add_run_to_train_data(env_aggregator)
            resets.append[run_done]
            observations.append(observation)
        is_done = self.train_runs <= len(self.train_data)
        observations = np.stack(observations)
        return (is_done, observations, resets)
    def _create_train_data_step_horizon(self, actions, value_estimate):
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        resets = []
        observations = []
        for action, value, env, env_aggregator in zip(actions, value_estimate, self.envs,
                                               self.envs_aggregator):
            observation, reward, run_done, _ = env.step(action)
            env_aggregator['action_list'].append(action)
            env_aggregator['value_list'].append(value)
            env_aggregator['observation_list'].append(observation)
            env_aggregator['reward_list'].append(reward)
            run_done = run_done or len(env_aggregator['reward_list'])
            self.train_samples
            if run_done:
                self._add_run_to_train_data(env_aggregator)
            resets.append[run_done]
            observations.append(observation)
        is_done = self.train_runs <= len(self.train_data)
        observations = np.stack(observations)
        return (is_done, observations, resets)

        
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
        for reverse_index in [-i for i in range(2,len(v_targs))]:
            v_targs[reverse_index] += self.value_gamma*v_targs[reverse_index+1]
        ### Implement gae - generalized advantage estimation
        ### shift values one to left and discount them by factor gamma
        shifted_discounted_value_estimations = np.asarray(env_aggregator['value_list']+[0])[1:] * self.value_gamma
        delta_t = rewards + shifted_discounted_value_estimations - np.asarray(env_aggregator['value_list'])
        gae_advantage = delta_t
        for reverse_index in [-i for i in range(2,len(delta_t))]:
            gae_advantage[reverse_index] += self.gae_lambda * self.value_gamma*gae_advantage[reverse_index + 1]
        ### put everything into the train_data
        run = {'action' : np.stack(env_aggregator['action_list']), 'advantage':
               gae_advantage,'v_targ' : v_targs, 'v_estimate' : np.stack(env_aggregator['value_list']),
               'reward' : rewards ,'observation': env_aggregator['observation_list']}
        self.train_data.append(run)
        for aggregator_list in env_aggregator.values():
            aggregator_list.clear()




'''
iteration_num = 10
env_name = #TODO
parallel_envs = 40
batch_size_data_creation = parallel_envs
batch_size_parameter_optimization = 100
lstm_unit_num = 128
value_gamma = 0.99
gae_lambda = 0.99
train_runs = 200
optimization_batches = 10
utility = None

for batch in range(batch_num):
    graph = tf.Graph()
    with graph.as_default():
        input_placeholder = tf.placeholder(tf.float32, shape = input_shape)
        print('creating')
        if batch == 0:
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        else:
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, custom_settings = True, 
                                                custom_bias = tf.initializers.constant(lstm_params[1]),
                                                custom_weights = tf.initializers.constant(lstm_params[0]))
        
        #print('building')
        lstm_results = []
        #print('setting state')
        state = lstm.zero_state(1,tf.float32)
        #print('unfolding')
        for t_step in range(input_length):
            output, state = lstm(input_placeholder[t_step, :,:], state)
            lstm_results.append(output)
        results = tf.stack(lstm_results)
        result_mean = tf.reduce_min(results)
        train_step = tf.train.AdamOptimizer().minimize(result_mean)
        #print('session')
        with tf.Session(graph = graph) as session:
            #print('initializing')
            session.run(tf.global_variables_initializer())
            #print('running')
            outcome_mean = session.run((result_mean), feed_dict = {input_placeholder :
                                                  predefined_input})
            print(outcome_mean)
            lstm_params = session.run(lstm.parameters)
        #print(lstm_params)
'''