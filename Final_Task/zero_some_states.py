import tensorflow as tf
from Final_Task.CustomCell import CustomBasicLSTMCell
import numpy as np


lstm_memory_size = 5
batch_size = 3
subsequence_length = 3



inputs  = tf.placeholder(tf.float32, shape=(batch_size, subsequence_length,
                                            lstm_memory_size))
sequences = tf.unstack(inputs, subsequence_length, axis=1)
print(sequences)
cell_state = tf.placeholder(tf.float32, shape=[batch_size, lstm_memory_size])
hidden_state = tf.placeholder(tf.float32, shape=[batch_size, lstm_memory_size])

input = tf.constant(6.0)


cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_memory_size)
zero_state = cell.zero_state(batch_size, tf.float32)
state = tf.nn.rnn_cell.LSTMStateTuple(c = cell_state, h = hidden_state)

outputs, state = tf.nn.static_rnn(cell, sequences, initial_state = state)
outputs = tf.reshape(tf.concat(outputs, 1), [batch_size, subsequence_length, lstm_memory_size])

my_inputs = np.random.rand(batch_size, subsequence_length, lstm_memory_size)
custom_cell_state = np.random.rand(batch_size, lstm_memory_size)
#print(custom_cell_state)
#print(my_inputs)

def zero_some_states(which_array, states):
    """
    changes the states of a BasicLSTM-StateTuple to 0 of some of the samples in the batch

    :param which_array: 1 means no reset and 0 resets the cell-state
    :param states: tf.nn.rnn_cell.LSTMStateTuple(c = cell_state,
                                                 h = hidden_state)
    """
    for i in which_array:
        if i == 1:
            pass
        elif i == 0:
            states.h[i] = 0
            states.c[i] = 0
        else:
            print("warning: there are only 1 and 0 allowed in the ident-array")

with tf.Session() as session:
    #print(outputs.eval())
    session.run(tf.global_variables_initializer())
    _state = session.run(zero_state)
    print('!!!', _state.h, _state.c)
    out_1, state_1 = session.run((outputs, state), feed_dict={inputs: my_inputs, cell_state: _state.c, hidden_state: _state.h})
    print("val1", np.mean(out_1))
    print(state_1.h)
    print(state_1.c)

    out_2, state_2 = session.run((outputs, state), feed_dict={inputs: out_1, cell_state: state_1.c, hidden_state: state_1.h})
    print("val2", np.mean(out_2))

    states_zero_some = zero_some_states([1, 0, 0], state_1)
    print(state_1.h)

    out_3, state_3 = session.run((outputs, state),
                                 feed_dict={inputs: out_1, cell_state: state_1.c, hidden_state: state_1.h})
    print("val3", np.mean(out_3))
    variables_names = [v.name for v in tf.global_variables()]
    #out_2, cell_s
    #print(hidden_state)
