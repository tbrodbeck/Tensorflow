# coding: utf-8

# # 3 Helper File
# 
# This week, you must use the python script from studip (04 svhn-helper.py) to read-in the data. The script converts the data into gray-scale images, splits the data into a training and validation dataset and provides a class to loop over the respective mini-batches. Be fair to the other groups and do not use the test dataset from the SVHN homepage to optimize your network.

# In[6]:


import os
import numpy as np
import scipy.io as scio
import time

t1 = time.time()
class SVHN():
    def __init__(self, directory="./"):
        self._directory = directory

        self._training_data = np.array([])
        self._training_labels = np.array([])
        self._test_data = np.array([])
        self._test_labels = np.array([])

        self._load_traing_data()
        # self._load_test_data()

        np.random.seed(0)
        samples_n = self._training_labels.shape[0]
        random_indices = np.random.choice(samples_n, samples_n // 10, replace=False)
        np.random.seed()

        self._validation_data = self._training_data[random_indices]
        self._validation_labels = self._training_labels[random_indices]
        self._training_data = np.delete(self._training_data, random_indices, axis=0)
        self._training_labels = np.delete(self._training_labels, random_indices)

    def _load_traing_data(self):
        self._training_data, self._training_labels = self._load_data("train_32x32.mat")

    def _load_test_data(self):
        self._test_data, self._test_labels = self._load_data("test_32x32.mat")

    def _rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _load_data(self, file):
        path = os.path.join(self._directory, file)

        mat = scio.loadmat(path)
        data = np.moveaxis(mat["X"], 3, 0)
        data = self._rgb2gray(data)
        data = data.reshape(data.shape + (1,))

        labels = mat["y"].reshape(mat["y"].shape[0])
        labels[labels == 10] = 0

        return data, labels

    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)

    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)

    # def get_test_batch(self, batch_size):
    #    return self._get_batch(self._test_data, self._test_labels, batch_size)

    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]

        if batch_size <= 0:
            batch_size = samples_n

        random_indices = np.random.choice(samples_n, samples_n, replace=False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]

    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n


# # 4 Investigate the data
# 
# As always, before you start, get an idea about how the data looks like, how the batches are structured and if the labels are assigned correctly.

# In[7]:


import tensorflow as tf
import matplotlib.pyplot as plt

svhn = SVHN()

# In[8]:


# retrieve training batch
images, label = next(svhn.get_training_batch(15))

# creating plot
fig, axs = plt.subplots(3, 5)
for i, ax in enumerate(np.reshape(axs, [-1])):
    ax.imshow(np.squeeze(images[i, :, :, :]), cmap='gray')
    ax.set_title(str(label[i]))
plt.suptitle('original data')
#plt.show()

# how the data looks like after we preprocessed it (cut off the sides)
fig, axs = plt.subplots(3, 5)
for i, ax in enumerate(np.reshape(axs, [-1])):
    ax.imshow(np.squeeze(images[i, :, 6:-6, :]), cmap='gray')  # this is how we will preprocess out data

    ax.set_title(str(label[i]))
plt.suptitle('preprocessed data')
#plt.show()


# # 5 Design a nework

# By now, you should have enough knowledge to design a network, to pick a suitable activation function for your hidden layers and the output layer, to chose a sophisticated gradient descent algorithm, to ﬁnd optimal hyperparameters and to implement the data-ﬂow graph in TensorFlow.

epochs = 8
mini_batch_size = 50
plot_step_size = 25
kernelsize_l1 = 5
kernelsize_l3 = 3

# defining input x as placeholder
x = tf.placeholder(tf.float32, shape=(mini_batch_size, 32, 32, 1))
labels = tf.placeholder(tf.int64, [mini_batch_size])

# LAYER ONE
# defining kernel of layer 1
kernel_l1 = tf.Variable(tf.truncated_normal(shape=(kernelsize_l1, kernelsize_l1, 1, 16), stddev=0.1))

# apply 1st layer convolution
featuremap_l1 = tf.nn.conv2d(x, kernel_l1, strides=[1, 1, 1, 1], padding="SAME")

# define bias layer one
bias_l1 = tf.Variable(tf.truncated_normal(shape=(32, 32, 16), stddev=0.1))

# Calculate neuron outputs by applying the activation function
activation_l1 = tf.nn.tanh(featuremap_l1 + bias_l1)

# LAYER TWO
# apply max pooling to outputs
pooling_l2 = tf.nn.max_pool(activation_l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# LAYER THREE
# defining kernel of layer 3
kernel_l3 = tf.Variable(tf.truncated_normal(shape=(kernelsize_l3, kernelsize_l3, 16, 32), stddev=0.1))

# apply 3rd layer convolution
featuremap_l3 = tf.nn.conv2d(pooling_l2, kernel_l3, strides=[1, 1, 1, 1], padding="SAME")

# define bias layer three
bias_l3 = tf.Variable(tf.truncated_normal(shape=(16, 16, 32), stddev=0.1))

# Calculate neuron outputs by applying the activation function
activation_l3 = tf.nn.tanh(featuremap_l3 + bias_l3)

# LAYER FOUR
# apply max pooling to outputs
pooling_l4 = tf.nn.max_pool(activation_l3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

# we reshape the feature maps
reshape_l4 = tf.reshape(pooling_l4, (mini_batch_size, 8*8*32))

# LAYER FIVE
weights_l5 = tf.Variable(tf.truncated_normal(shape=(8*8*32, 512), stddev=0.1))
bias_l5 = tf.Variable(tf.truncated_normal(shape=[512], stddev=0.1))
drive_l5 = tf.nn.tanh(tf.matmul(reshape_l4, weights_l5) + bias_l5)

# LAYER SIX
weights_l6 = tf.Variable(tf.truncated_normal(shape=(512, 10), stddev=0.1))
bias_l6 = tf.Variable(tf.truncated_normal(shape=[10], stddev=0.1))
# output of our NN
y = tf.matmul(drive_l5, weights_l6) + bias_l6

# ERROR CALCULATION
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y))
optimizer = tf.train.AdamOptimizer(1e-3)
training_step = optimizer.minimize(cross_entropy)

accuracy = tf.equal(tf.argmax(tf.nn.softmax(y), 1), labels)
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

train_c_e = []
train_acc = []

val_c_e = []
val_acc = []

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    step = 0

    for epoch in range(epochs):
        for imgs, lbls in svhn.get_training_batch(mini_batch_size):

                training_cross_entropy, training_accuracy, step = session.run(
                    [cross_entropy, accuracy, training_step],
                    feed_dict={x: imgs, labels: lbls}
                )

# EVALUATION
# Plot initialization
train_c_e = []
train_acc = []

val_c_e = []
val_acc = []

# Session
session = tf.Session()
session.run(tf.global_variables_initializer())

# EVALUATION
# Plot initialization
plt.ion()
figs, (ax1, ax2) = plt.subplots(2, 1)
plt.title("Evaluation during training...")

total_steps = 0

for epoch in range(epochs):
    print('Epoch Number:', epoch)
    for imgs, lbls in svhn.get_training_batch(mini_batch_size):
        training_cross_entropy, training_accuracy, _ = session.run(
            [cross_entropy, accuracy, training_step],
            feed_dict={x: imgs, labels: lbls}
        )
        total_steps += 1

        train_c_e.append(training_cross_entropy)
        train_acc.append(training_accuracy)

        if total_steps % plot_step_size == 0:
            val_accuracy, val_cross_ent = session.run([accuracy, cross_entropy],
                                                      feed_dict={x: svhn._validation_data,
                                                                 labels: svhn._validation_labels})
            print(val_accuracy)
            val_c_e.append(val_cross_ent)
            val_acc.append(val_accuracy)

            plt.sca(ax1)
            ax1.cla()
            plt.sca(ax2)
            ax2.cla()

            ax1.plot(range(0, total_steps), train_c_e, 'b-')
            ax1.set_ylabel('Cross Entropy {:1.3f}'.format(train_c_e[-1]))
            ax1.set_xlabel('Training step {:d}'.format(total_steps))
            ax1.legend()

            ax2.plot(range(0, total_steps), train_acc, 'b-')
            ax2.set_ylabel('Accuracy {:1.3f}'.format(train_acc[-1]))
            ax2.set_xlabel('Training step {:d}'.format(total_steps))
            ax2.legend()

            ax1.plot(range(0, total_steps, plot_step_size), val_c_e, 'o', c='orange')
            ax1.set_ylabel('Cross Entropy {:1.3f}'.format(val_c_e[-1]))
            ax1.set_xlabel('Trainingsstep {:d}'.format(total_steps))

            ax2.plot(range(0, total_steps, plot_step_size), val_acc, 'o', c='orange')
            ax2.set_ylabel(('Accuracy {:1.3f}'.format(val_acc[-1])))
            ax2.set_xlabel('Trainingstep {:d}'.format(total_steps))

        figs.canvas.draw()

plt.ioff()
plt.title('Evaluation during training completed')
plt.show()

# # 6 Preprocessing

# Make sure to include any pre-processing into the computational graph of your network, otherwise it won’t be applied to the test data. You are free to augment the training data.


# # 7 Save the weights

# Use the Saver class to save the progress of your training data. Do not forget to include the ﬁnal checkpoint ﬁle in your submission.


# # 9 Evaluation

# Please add the code of 04 svhn-evaluation.py, which can be found on studip, to the end of your ipython notebook and add any further placeholder values (i.e. a dropout rate) that you would like to feed to your network. This part of your submission will be used to evaluate the performance of your network.

'''
with tf.Session() as session:
    saver = tf.train.Saver()
    saver.restore(session,  tf.train.latest_checkpoint("./weights/"))
    
    test_accuracy = 0
    for step, (images, labels) in enumerate(svhn.get_test_batch(300)):
        test_accuracy += session.run(
            accuracy,
            feed_dict = {x: images, desired: labels}
        )
    
print("Test Accuracy: " + str(test_accuracy / step))
'''