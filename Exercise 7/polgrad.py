import gym
import numpy as np
import tensorflow as  tf
import matplotlib.pyplot as plt

def compute_gradients_and_rewards(gradients, rewards, discount_factor, gradient_list_pointer):
    seq_length = len(rewards)
    #compute the respective reward values
    rewards = np.asarray(rewards)
    last_reward = 0
    for i in np.arange(seq_length):
        index_from_end = (i+1)*(-1)
        rewards[index_from_end] = (discount_factor * last_reward) + rewards[index_from_end]
        last_reward = rewards[index_from_end]
    rewards -= np.mean(rewards)
    rewards = rewards / np.std(rewards)


    for i in np.arange(seq_length):
        for gradient_index in np.arange(len(gradients[i])):
            gradients[i][gradient_index] = gradients[i][gradient_index] * rewards[i]
            #print(gradients[i][gradient_index])

    variable_length = len(gradients[0])
    summed_gradient_list = []
    for i in np.arange(variable_length):
        summed_gradient_list.append(np.zeros(np.shape(gradients[0][i])))
        for j in np.arange(seq_length):
            summed_gradient_list[i] += gradients[j][i]

    grad_dict = {}
    for i in np.arange(len(gradient_list_pointer)):
        grad_dict[gradient_list_pointer[i]] = summed_gradient_list[i]
    return grad_dict, rewards

    grad_dict = {}
    for i, placeholder in enumerate(gradient_list_pointer):
        grad_dict[placeholder] = gradients[i]
    return grad_dict, rewards

####parameters####
episodes = 500
episode_length = 200
learning_rate = 0.01
discount_factor = 0.97
batch_size = 15
show_sample_factor = 20

#### the graph ####
x = tf.placeholder(tf.float32, shape = (1,4))
#FFLayer
w01 = tf.Variable(tf.random_normal(shape = (4,20), stddev = 0.02))
b01 = tf.Variable(tf.zeros(shape = (1,20)))
l01 = tf.nn.relu(tf.add(tf.matmul(x,w01), b01))

## Outlayer
w02 = tf.Variable(tf.random_normal(shape = (20,1), stddev = 0.02))
b02 = tf.Variable(tf.zeros(shape = (1,1)))
prob01 = tf.nn.sigmoid(tf.add(tf.matmul(l01,w02), b02))
prob02 = tf.subtract(1.0, prob01)

probs = tf.log(tf.concat([prob01, prob02],1))

#logs
action = tf.multinomial(probs,1)[0][0]

log_likelihood = probs[:, tf.to_int32(action)]

### from Lukas code

# Create optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)

# Compute gradients, returns a list of gradient variable tuples
gradients_and_variables = optimizer.compute_gradients(log_likelihood)

# Extract gradients and inverse the sign of gradients
# (compute_gradients returns inverted gradients for minimization)
gradients = [gradient_and_variable[0] * -1 for gradient_and_variable in gradients_and_variables]

# Retrieve and modify the gradients from within a session
# Create placeholders for modified gradients
gradient_placeholders = []
for gradient in gradients:
    gradient_placeholders.append(tf.placeholder(tf.float32, gradient.shape))

# Apply gradients
trainable_variables = tf.trainable_variables()
training_step = optimizer.apply_gradients(zip(gradient_placeholders, trainable_variables))

#start session, deploy graph
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    #get ai_gym environment
    env = gym.make('CartPole-v0')

    #lists for buffering results
    gradient_dictionary_list = []
    success_list = []
    #go over all the episodes
    for episode in np.arange(episodes):
        observation = env.reset()
        #reset the buffers
        gradient_list = []
        reward_list = []
        #go through all the steps
        for step in np.arange(episode_length):
            #get observation and compute action
            observation = observation[np.newaxis]
            selected_action, extracted_gradient= session.run([action, gradients], feed_dict = {x: observation})
            observation, reward, done, info = env.step(selected_action)
            #keep track of rewards and gradients
            gradient_list.append(extracted_gradient)
            reward_list.append(reward)

            #check whether we should show this episode
            if episode%show_sample_factor == 0:
                env.render()
            #break loop when we are done
            if done:
                break

        # get weighted gradients and discount the rewards
        gradient_dictionary, discounted_rewards = compute_gradients_and_rewards(gradient_list, reward_list, discount_factor, gradient_placeholders)
        gradient_dictionary_list.append(gradient_dictionary)
        success_list.append(len(discounted_rewards))

        #only update the model when we have enough steps for a full batch of updating
        if episode % batch_size == 0:
            for feed_dictionary in gradient_dictionary_list:
                _ = session.run(training_step, feed_dict = feed_dictionary)
            gradient_dictionary_list = []

    #plot results
    plt.plot(np.asarray(success_list))
    plt.show()
