
# coding: utf-8

# # Reinforcement Learning: Policy Gradients
# 
# Homework assignments are mandatory. In order to be granted with 8 ECTS and your grade, you must pass 9 out of 10 homeworks. Upload your solution as zip archive until Saturday 13th January 23:59 into the public Homework Submissions folder on studip. The archive should contain your solution as iPython notebook (.ipynb) and as HTML export. Name the archive ﬁle as <your group id > policy-gradients.zip.
# 
# Further, you have to correct another group’s homework until Monday 15th January 23:59. Please follow the instructions in the rating guidelines on how to do your rating.
# 
# If you encounter problems, please do not hesitate to send your question or concern to lbraun@uos.de. Please do not forget to include [TF] in the subject line.
# 
# # 1 Introduction
# 
# In this week’s task, we are going to implement a network that learns to balance a pole on a cart. The network will learn how to map an observation to an appropriate action.
# # 2 Data, OpenAI Gym
# Install the OpenAI Gym pip package (pip instsall gym). Create a CartPolev0 environment and create a loop that can retrieve an observation and selects and executes a random action, until the episode is done. Create a loop around that loop, such that you can run the environment for several episodes. Render the environment every N episodes and keep in mind, that rendering does slow down the learning drastically, since if the process is not visualized, it does not restrict the frame rate and runs as fast as possible.
# 
# Observation 
# 
# Each observation is a four-dimensional array: [cart position, cart velocity, pole angle, pole velocity at tip].
# 
# Action 
# 
# space At each time-step, the agent has to decide to either push the cart to the left or the right.
# 
# Reward 
# 
# The agent receives a +1 reward for each time-step until the episode terminates.
# 
# Episode Termination
# 
# The episode terminates, if either the pole angle is more than +/- 12 degree, the cart moved out of the display or if the episode length is greater than 200. Hence, the maximum reward that can be reached is 200.

# In[18]:


import gym
import numpy as np
import tensorflow as  tf

#### the graph ####
x = tf.placeholder(tf.float32, shape = (1,4))
# hidden layer
w01 = tf.Variable(tf.random_normal(shape = (4,8), stddev = 0.1))
b01 = tf.Variable(tf.random_normal(shape = (1,8), stddev  = 0.1))
l01 = tf.nn.relu(tf.add(tf.matmul(x,w01), b01))

# output layer
w02 = tf.Variable(tf.random_normal(shape = (8,1), stddev = 0.1))
b02 = tf.Variable(tf.random_normal(shape = (1,1), stddev  = 0.1))
prob01 = tf.nn.sigmoid(tf.add(tf.matmul(l01,w02), b02))
prob02 = tf.subtract(1.0, prob01)

# Create log likelihoods from a probability distribution over actions
log_likelihoods = tf.log(tf.concat([prob01, prob02],1))

# sample action from that distribution
action = tf.multinomial(log_likelihoods,1)[0][0]
print(action)

# select value that corresponds to selected action
log_likelihood = log_likelihoods[:, tf.to_int32(action)]
print(log_likelihood)


####parameters####

episodes = 100
episode_length = 200
learning_rate = 0.1
discount = 0.9


####calculate gradient####

# create optimizer

optimizer = tf.train.AdamOptimizer(learning_rate)


# compute gradients, returns a list of gradien variable tuples
gradients_and_variables = optimizer.compute_gradients(log_likelihood)

# Extract gradients and inverse the sign of gradients
# (compute_gradients returns inverted gradients for minimization)
gradients = [gradients_and_variable[0] * -1 for gradients_and_variable in gradients_and_variables]


####retrieve and modify the gradients from within a session####

# create placeholders for modified gradients
gradient_placeholders = []
for gradient in gradients:
    gradient_placeholders.append(tf.placeholder(tf.float32, gradient.shape))

# apply gradients
trainable_variables = tf.trainable_variables()
training_step = optimizer.apply_gradients(zip(gradient_placeholders, trainable_variables))


####execution####

with tf.Session() as session:
    
    # initialize variables
    session.run(tf.global_variables_initializer())
    
    # create environment
    env = gym.make('CartPole-v0')
    
    # for each episode (epoch)
    for episode in np.arange(episodes):
        
        # Reset Game
        observation = env.reset()

        # buffer our rewards
        rewardList = []

        # buffer our gradients
        gradientList = []
        
        for step in np.arange(episode_length):
            
            observation = observation[np.newaxis]
            #print(observation)
            
            # Create an action (i.e. sample from network)
            _action, _gradient = session.run([action, gradients], feed_dict = {x: observation})
            
            # Execute action and receive the corresponding reward 
            # and a new observation
            observation, reward, done, _ = env.step(_action)

            # buffering
            rewardList.append(reward)
            gradientList.append(_gradient)

            #print(reward)
            #print(_gradient)
            
            if done:
                break
                
            # render the visualization
            #if episodes%10:
                #env.render()

        # calculate discounted rewards
        discounted = 0.
        for index, item in enumerate(rewardList):
            discounted = (discounted + item) * discount
        print(discounted)


