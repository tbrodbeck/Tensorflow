import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from utilities import TrainingUtil
from policy import Policy


''' Hyperparameters '''

# number of iterations for the whole algorithm
iteration_num = 1000
# name of the openaigym used
env_name = 'Ant-v1'
# size of the observation
observation_size = gym.make(env_name).observation_space.shape[0]
# size of an action (the output for our policy has to be twice as big, as we
# have to model a probability density function pdf over it)
action_size = gym.make(env_name).action_space.shape[0]
# how many environments should be used to generate train_data at once
parallel_envs = 100
# batch size for network in creating training data
batch_size_data_creation = parallel_envs
# size of a minibatch in in optimization
batch_size_parameter_optimization = 50
# amount of epochs to train over one set of training_data
optimization_epochs = 5
# size of the lstm cell
lstm_unit_num = 128
# gamma value for discounting rewards for value function
value_gamma = 0.99
# lambda value for generalized advantage estimator gae values
gae_lambda = 0.95
# amount of training runs to assemble for one training-optimization iteration
train_runs = 200
# length of one training run, THIS IS NOT USED IN 'runs'
horizon = 30
# length of the subsequences we will train on
training_sequence_length = horizon
assert training_sequence_length <= horizon
# train mode, either 'horizon' or 'runs'
train_mode = 'horizon'
# learn_rate
learn_rate = 0.005
# epsilon for l_clip loss function
epsilon = 0.2
# c1, hyperparameter factor for weighting l_value loss
c1 = 0.05
# c2, hyperparameter factor for weighting l_exploration loss
c2 = 0.01


''' Training '''

# This utility class saves weights and keeps track of the training_data
utility = TrainingUtil(None, parallel_envs, gae_lambda, value_gamma,
                       env_name, train_runs, train_mode, horizon)
# keeping track of training success
rewards_list = []
loss_list = [[], [], [], []]
plt.ion()
# for dictionary access
keys = ['lstm_weights', 'lstm_bias', 'embedding_weights',
        'embedding_bias', 'value_readout_weights',
        'value_readout_bias', 'action_alpha_readout_weights',
        'action_alpha_readout_bias',
        'action_beta_readout_weights', 'action_beta_readout_bias']

print('Start training!')

for iteration in range(iteration_num):
    # for plotting
    loss_iteration_list = [[], [], [], []]
    # deploy a new graph for every new training_iteration, minimizing the
    # trash left over in our RAM
    graph = tf.Graph()
    # First we build the train_data_set
    # The old network generates train_samples
    train_data_network = None


    ''' Retrieving Training Samples'''

    with graph.as_default():
        step = 0
        done = False

        # create or retrieve corresponding network for new iteration
        if iteration == 0:
            train_data_network = Policy(lstm_unit_num, observation_size,
                                         'iteration' + str(iteration) + 'train_data_generation',
                                        action_size, batch_size_data_creation)
        else:
            train_data_network = Policy(lstm_unit_num, observation_size,
                                         'iteration' + str(iteration) + 'train_data_generation', action_size,
                                        batch_size_data_creation, utility.weights)

        while not done:
            # generate one step of the policy
            value, alpha, beta, action = train_data_network.step(
                'unfold_iteration' + str(iteration) + 'step' + str(step),
                step)
            with tf.Session(graph=graph) as session:
                session.run(tf.global_variables_initializer())
                value, alpha, beta, action = session.run((value, alpha, beta, action),
                                                         feed_dict={
                                                             train_data_network.observation: utility.get_observation()})
                is_done, resets = utility.create_train_data_step(action, value, alpha, beta)

                print('Sampling: Iteration:' + str(iteration) +' Step:' + str(step))

                if train_mode is 'horizon' and step % horizon == 0:
                    train_data_network.reset_states()
                step += 1
                done = is_done

        if iteration == 0:
            with tf.Session(graph=graph) as session:
                session.run(tf.global_variables_initializer())
                parameters = session.run(train_data_network.network_parameters())
                weights = {}
                # keys are defined in the hyperparameter list
                for parameter, key in zip(parameters, keys):
                    weights[key] = parameter
                train_data = utility.train_data
                utility = TrainingUtil(weights, parallel_envs, gae_lambda, value_gamma, env_name, train_runs,
                                       train_mode, horizon)
                utility.train_data = train_data


    ''' Optimization step '''

    # Now we got the trian_data
    graph = tf.Graph()
    with graph.as_default():
        optimizing_network = Policy(lstm_unit_num,
                                    observation_size, 'iteration' + str(iteration) + 'optimization',
                                    action_size,
                                    batch_size_parameter_optimization,
                                    utility.weights, learn_rate=learn_rate)

        ### and now we have to implement the training procedure
        for epoch in range(optimization_epochs):
            # this is messy, might still work
            used_samples = train_runs - (train_runs % batch_size_parameter_optimization)
            train_sample_plan = np.reshape(np.arange(used_samples),
                                           (int(used_samples / batch_size_parameter_optimization),
                                            batch_size_parameter_optimization))
            np.random.shuffle(train_sample_plan)
            train_sample_plan = train_sample_plan.tolist()
            train_data = utility.train_data

            for count, sample in enumerate(train_sample_plan):
                print('Optimization: Iteration:' + str(iteration) + ' Epoch:' + str(epoch) + ' Run:' + str(count))

                alpha = np.stack([train_data[i]['alpha'] for i in sample], axis=1)
                beta = np.stack([train_data[i]['beta'] for i in sample], axis=1)
                advantages = np.stack([train_data[i]['advantage'] for i in sample], axis=1)
                v_targ = np.stack([train_data[i]['v_targ'] for i in sample], axis=1)
                action = np.stack([train_data[i]['action'] for i in sample], axis=1)
                observation = np.stack([train_data[i]['observation'] for i in sample], axis=1)

                train_step, loss = optimizing_network.optimize(
                    'iteration' + str(iteration) + 'optimizationepoch' + str(epoch), training_sequence_length, epsilon,
                    c1, c2)

                with tf.Session(graph=graph) as session:
                    # can not preconstruct initializer, as new variables are added
                    session.run(tf.global_variables_initializer())
                    _, loss = session.run((train_step, loss), feed_dict=
                    {optimizing_network.alpha: alpha,
                     optimizing_network.beta: beta,
                     optimizing_network.gae_advantage: advantages,
                     optimizing_network.target_value: v_targ,
                     optimizing_network.action: action,
                     optimizing_network.optimization_observation:
                         observation})

                    for list, losses in zip(loss_iteration_list, loss):
                        list.append(losses)


        ''' Plotting '''

        # retrieve data
        rewards_list.append(utility.get_average_reward())
        for list, losses in zip(loss_list, loss_iteration_list):
            list.append(np.mean(losses))

        print('rwrd:', rewards_list)
        print('loss:',loss_list)

        # plotting each x step
        plot_step = 15
        if ((iteration % plot_step) - 1 == 0):
            # plot of reward
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.plot(rewards_list, label='Avarage Reward')
            ax.legend()
            # plot of loss functions
            fig2 = plt.figure(figsize=(10, 10))
            ax1 = fig2.add_subplot(221)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('loss_complete')
            ax1.plot(loss_list[0])
            ax2 = fig2.add_subplot(222)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('loss_clip')
            ax2.plot(loss_list[1])
            ax3 = fig2.add_subplot(223)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('loss_value')
            ax3.plot(loss_list[2])
            ax4 = fig2.add_subplot(224)
            ax4.set_xlabel('Step')
            ax4.set_ylabel('loss_explore')
            ax4.plot(loss_list[3])
            # for plotting while continuing running program
            plt.draw()
            plt.pause(0.5)

        test = gym.make(env_name)
        test.step()
        test.render()

        ''' Save the trained parameters '''

        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            parameters = session.run(optimizing_network.network_parameters())
            weights = {}
            # keys are defined in the hyperparameter list
            for parameter, key in zip(parameters, keys):
                weights[key] = parameter
            utility = TrainingUtil(weights, parallel_envs, gae_lambda, value_gamma,
                                   env_name, train_runs, train_mode,
                                   horizon)

# show final plot
plt.ioff()
plt.show()

