import gym
import numpy as np
import tensorflow as  tf

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

####parameters####
episodes = 100
episode_length = 200
learn_rate = 0.1
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    env = gym.make('CartPole-v0')
    for episode in np.arange(episodes):
        observation = env.reset()
        for step in np.arange(episode_length):
            observation = observation[np.newaxis]

            print(observation)
            observation, reward, done, info = env.step(session.run(action, feed_dict = {x: observation}))

            if done:
                break
            env.render()
