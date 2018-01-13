import gym
import numpy as np


#### the graph ####
x = tf.Placeholder(tf.float32, shape = (1,4))
#FFLayer
w01 = tf.Variable(tf.random_normal(shape = (4,8), stddev = 0.1))
b01 = tf.Variable(tf.random_normal(shape = (1,8)), stddev  = 0.1)
l01 = tf.nn.relu(tf.add(tf.matmul(x,w01), b01))

## Outlayer
w02 = tf.Variable(tf.random_normal(shape = (8,1), stddev = 0.1))
b02 = tf.Variable(tf.random_normal(shape = (1,1)), stddev  = 0.1)
prob01 = tf.nn.sigmoid(tf.add(tf.matmul(l01,w02), b02))
prob02 = tf.subtract(1, prob01)

#logs
action = tf.multinomial(tf.log(tf.concat([prob01, prob02],1)))

####parameters####
episodes = 100
episode_length = 200
learn_rate = 0.1
with tf.session() as sess:
    sess.run()
    env = gym.make('CartPole-v0')
    for episode in np.arange(episodes):
        env.reset()
        for step in np.arange(episode_length):




env =
env.reset()
for i in np.arange(0,200):
    observation, reward, done, info = env.step(np.random.randint(0,2))
    if done:
        break
    env.render()
