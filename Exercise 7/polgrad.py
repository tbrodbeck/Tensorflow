import gym
import numpy as np
import tensorflow as  tf

####parameters####
episodes = 100
episode_length = 200
learning_rate = 0.1
discount_factor = 0.9

#### the graph ####
x = tf.placeholder(tf.float32, shape = (1,4))
#FFLayer
w01 = tf.Variable(tf.random_normal(shape = (4,8), stddev = 0.1))
b01 = tf.Variable(tf.random_normal(shape = (1,8), stddev  = 0.1))
l01 = tf.nn.relu(tf.add(tf.matmul(x,w01), b01))

## Outlayer
w02 = tf.Variable(tf.random_normal(shape = (8,1), stddev = 0.1))
b02 = tf.Variable(tf.random_normal(shape = (1,1), stddev  = 0.1))
prob01 = tf.nn.sigmoid(tf.add(tf.matmul(l01,w02), b02))
prob02 = tf.subtract(1.0, prob01)

#logs
action = tf.multinomial(tf.log(tf.concat([prob01, prob02],1)),1)[0][0]

### from Lukas code

# Create optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)

# Compute gradients, returns a list of gradient variable tuples
gradients_and_variables = optimizer.compute_gradients(tf.log(prob01))

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

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    env = gym.make('CartPole-v0')
    for episode in np.arange(episodes):
        observation = env.reset()
        gradientList = []
        rewardList = []
        for step in np.arange(episode_length):
            observation = observation[np.newaxis]
            selected_action, extracted_gradient = session.run([action, gradients], feed_dict = {x: observation})
            observation, reward, done, info = env.step(selected_action)
            #keep track of rewards and gradients
            gradientList.append(extracted_gradient)
            print('reward:')
            print(reward)
            if done:
                break

            env.render()
