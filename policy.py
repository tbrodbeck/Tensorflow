import tensorflow as tf
from custom_lstm_cell import CustomBasicLSTMCell

class Policy:
    def __init__(self, num_units, observation_size, name, action_size,
                 batch_size, weights=None, learn_rate=None):
        """
        We define the network here.

        :param num_units: int, lstm_unit_num
        :param observation_size: int, size of respective env.observation_space
        :param name: str
        :param action_size: int, size of respective env.action_space
        :param batch_size: int
        :param weights: parameters of policy
        :param learn_rate: if given, the network is in optimization mode, uses
        learn_rate as learning_rate
        """
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

            if learn_rate is not None:
                self.optimizer = tf.train.AdamOptimizer(learn_rate)

    def reset_states(self):
        self.state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

    def step(self, name, step):
        """
        we run the policy for one step and retrieve the observable data

        :param name: str, step name
        :param step: int
        :return: value, action_alpha, action_beta, action
        """
        with tf.variable_scope(name):
            # TODO maybe simply saving weights in a variable would be an option

            lstm_out, self.state = self.lstm(self.embedding, self.state)
            value = tf.add(tf.matmul(lstm_out, self.value_readout_weights),
                           self.value_readout_bias)
            action_alpha = tf.add(tf.nn.relu(tf.add(tf.matmul(lstm_out,
                                                              self.action_alpha_readout_weights),
                                                    self.action_alpha_readout_bias)), tf.constant(1,
                                                                                                  dtype
                                                                                                  =
                                                                                                  tf.float32))
            action_beta = tf.add(tf.nn.relu(tf.add(tf.matmul(lstm_out,
                                                             self.action_beta_readout_weights),
                                                   self.action_beta_readout_bias)), tf.constant(1,
                                                                                                dtype
                                                                                                =
                                                                                                tf.float32))
            action = tf.distributions.Beta(action_alpha, action_beta).sample()
            return value, action_alpha, action_beta, action

    def multiply_t(self, x, y, sequence_length):
        """
        This function computes the tf.matmul operation for some
        time-series input of size [t_steps, batch_size, vector_size] and a
        weight matriy of size [vector_size, output_size]
        """
        return tf.matmul(x, tf.stack([y for _ in range(sequence_length)], 0))

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
                                        shape=(sequence_length, self.batch_size, self.action_size))
            self.beta = tf.placeholder(dtype=tf.float32,
                                       shape=(sequence_length, self.batch_size, self.action_size))
            self.action = tf.placeholder(dtype=tf.float32,
                                         shape=(sequence_length, self.batch_size, self.action_size))
            self.gae_advantage = tf.placeholder(dtype=tf.float32,
                                                shape=(sequence_length, self.batch_size))
            self.target_value = tf.placeholder(dtype=tf.float32,
                                               shape=(sequence_length, self.batch_size))
            self.optimization_observation = tf.placeholder(dtype=tf.float32,
                                                           shape=
                                                           (sequence_length,
                                                            self.batch_size,
                                                            self.observation_size))

            self.state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
            embedding = tf.nn.relu(tf.add(self.multiply_t(self.optimization_observation,
                                                          self.embedding_weights,
                                                          sequence_length),
                                          tf.squeeze(self.embedding_bias)))
            lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm, embedding,
                                                     initial_state=
                                                     self.state, time_major=
                                                     True)
            val_pred = tf.add(self.multiply_t(lstm_out,
                                              self.value_readout_weights,
                                              sequence_length),
                              tf.squeeze(self.value_readout_bias))
            alpha_pred = tf.add(tf.nn.relu(tf.add(self.multiply_t(lstm_out,
                                                                  self.action_alpha_readout_weights,
                                                                  sequence_length),
                                                  tf.squeeze(self.action_alpha_readout_bias))),
                                tf.constant(1, dtype=tf.float32))
            beta_pred = tf.add(tf.nn.relu(tf.add(self.multiply_t(lstm_out,
                                                                 self.action_beta_readout_weights,
                                                                 sequence_length),
                                                 tf.squeeze(self.action_beta_readout_bias))),
                               tf.constant(1, dtype=tf.float32))
            policy_new = tf.distributions.Beta(alpha_pred, beta_pred)
            policy_old = tf.distributions.Beta(self.alpha, self.beta)
            prob_new = policy_new.prob(tf.squeeze(self.action))
            prob_old = policy_old.prob(tf.squeeze(self.action))
            entropy = policy_new.entropy()
            entropy_product = tf.reduce_prod(entropy, axis=2)
            prob_product_old = tf.reduce_prod(prob_old, axis=2)
            prob_product_new = tf.reduce_prod(prob_new, axis=2)
            l_explore = entropy_product
            l_value = tf.square(tf.subtract(tf.squeeze(val_pred), self.target_value))
            prob_ratio = tf.divide(prob_product_new, prob_product_old)
            l_clip = tf.minimum(tf.multiply(prob_ratio, self.gae_advantage),
                                tf.multiply(tf.clip_by_value(prob_ratio, 1 - epsilon, 1 + epsilon), self.gae_advantage))
            loss_complete = tf.add(tf.subtract(l_clip,
                                               tf.multiply(tf.constant(c1, dtype=tf.float32),
                                                           l_value)),
                                   tf.multiply(tf.constant(c2, dtype=tf.float32), l_explore))
            inverted_loss = tf.multiply(tf.constant(-1, dtype=tf.float32),
                                        l_clip)
            learn_step = self.optimizer.minimize(tf.reduce_mean(inverted_loss))
            self.state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
            losses = [tf.reduce_mean(loss_complete), tf.reduce_mean(l_clip),
                      tf.reduce_mean(l_value), tf.reduce_mean(l_explore)]
            return learn_step, losses

    def network_parameters(self):
        """
        :return: all trainable parameters of the network
        """
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