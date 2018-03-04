import tensorflow as tf
import numpy as np

input_size = 5
batch_size = 2
max_length = 1

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 4)

# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, max_length, input_size])
output, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    result = sess.run([output], feed_dict={data: np.ones((batch_size, max_length, input_size))})

    print(result)
    print(result[0].shape)

    for v in tf.trainable_variables():
        print(v.name)
        print(dir(v))