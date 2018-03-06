import random
import gym
import numpy as np

class TrainingUtil:

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
                                         : [], 'alpha_list': [], 'beta_list': []})
            self.observation.append(self.envs[elem].reset())

        self.gae_lambda = gae_lambda
        self.value_gamma = value_gamma
        """
        trainmode switches between:
        'runs' : using @parallel_train_units environments, create
        @train_runs runs, each with possibly different run length
        in this mode, train_samples is not used
        'horizon': create train_runs runs of length
        @train_samples. train_runs should be multiple of
        parallel_train_units in this case
        """
        assert train_mode in ['runs', 'horizon']
        self.train_mode = train_mode
        self.train_runs = train_runs
        self.horizon = horizon

    def get_average_reward(self):
        """
        :return: avarage reward of the train_data
        """
        rewards_acc = []
        for sample in self.train_data:
            rewards_acc.append(sample['reward'])
        return np.mean(np.stack(rewards_acc))

    def create_train_data_step(self, actions, value_estimate, alpha, beta):
        """
        actions is numpy array of parallel_train_units, action_size
        this calls step(actions[respective_env_num]) on each env
        returns (is_done, observations, resets):
        is_done is true iff train_runs is reached
        resets is list of size(parallel_train_units), detailing, which
        train_units are reset (and should have their cellstate cleared
        respectively), where 1: should be reset, 0: should not be reset
        """
        actions = np.split(actions, len(actions[:, 0]))
        value_estimate = np.split(value_estimate, len(value_estimate))
        if self.train_mode == 'runs':
            return self._create_train_data_step_runs(actions, value_estimate,
                                                     alpha, beta)
        elif self.train_mode == 'horizon':
            return self._create_train_data_step_horizon(actions, value_estimate, alpha, beta)

    def _create_train_data_step_runs(self, actions, value_estimate, alpha, beta):
        """
        if train_mode is run
        """
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        self.observation = []
        resets = []
        for action, value, env, env_aggregator, alpha_val, beta_val in zip(actions, value_estimate, self.envs,
                                                                           self.envs_aggregator, alpha, beta):

            env_action = action * 2
            env_action = env_action - 1
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

    def _create_train_data_step_horizon(self, actions, value_estimate, alpha, beta):
        """
        if train_mode is horizon
        """
        assert len(actions) == len(self.envs)
        assert len(actions) == len(value_estimate)
        resets = []
        self.observation = []
        for action, value, env, env_aggregator, alpha_val, beta_val in zip(actions, value_estimate, self.envs,
                                                                           self.envs_aggregator, alpha, beta):
            env_action = action * 2
            env_action = env_action - 1
            new_observation, reward, run_done, _ = env.step(action)
            self.observation.append(new_observation)
            env_aggregator['alpha_list'].append(alpha_val)
            env_aggregator['beta_list'].append(beta_val)
            env_aggregator['action_list'].append(np.squeeze(action))
            env_aggregator['value_list'].append(np.squeeze(value))
            env_aggregator['observation_list'].append(new_observation)
            env_aggregator['reward_list'].append(reward)
            run_done = run_done or len(env_aggregator['reward_list']) >= self.horizon

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
        shifted_discounted_value_estimations = np.asarray(env_aggregator['value_list'] + [0])[1:] * self.value_gamma
        delta_t = rewards + shifted_discounted_value_estimations - np.asarray(env_aggregator['value_list'])
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
               'beta': np.stack(env_aggregator['beta_list'])}
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

        for run_nr, run in enumerate(self.train_data):
            # chose on training occurance
            training = run['action']
            print(len(training))
            assert (len(training) > length)

            # amount of samples we have to skip
            cutout = len(training) % length
            print(cutout)
            print(len(training))

            # by chance choose the front-part of the trajectory
            if cutout == 0:
                for index in range(len(training) / length):
                    beginn = index * length
                    ending = index * length + length - 1
            if bool(random.getrandbits(1)):
                trajectory = training[:(len(training)) - cutout]
                print(trajectory)
                amount = len(trajectory) // length
                print(91, amount)
                assert amount > 0
                for index in range(amount):
                    beginn = index * length
                    ending = index * length + length
            # or choose the back-part of the trajectory
            else:
                trajectory = training[cutout - 1:]
                print(trajectory)
                amount = len(trajectory) // length
                print(99, amount)
                for index in range(amount):
                    beginn = index * length + cutout
                    ending = index * length + length + cutout

            dicts.append({'run': run_nr, 'beg': beginn, 'end': ending})

        random.shuffle(dicts)
        return dicts
