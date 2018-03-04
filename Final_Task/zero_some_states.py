import tensorflow as tf
import numpy as np


lstm_memory_size = 5
batch_size = 3
subsequence_length = 3

'''
def zero_some(bool_array):
    arr = []
    for step, zero in enumerate(bool_array):
        if zero:
            arr.append(0)

        elif zero:
            arr.append(1)
        else:
            print("warning: there are only 1 and 0 allowed in the ident-array")
    return tf.
 '''

'''
def zero_some(batch_array, lstm_memory_s):
    tensors = []
    for step, action in enumerate(batch_array):
        if action:
            tensors.append(tf.ones(lstm_memory_size))

        elif action:
            tensors.append(tf.zeros(lstm_memory_size))
        else:
            print("warning: there are only 1 and 0 allowed in the ident-array")
        if step > 1:
            tensors[step] = tf.concat(values={tensors[step - 1], tensors[step]}, axis=0)
    return tensors[len(batch_array)]
'''
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
print('state!!', state.c)
print('2', state)
zero_one = tf.placeholder(tf.float32, shape=[3])
print(zero_one)
zero_one = tf.transpose(zero_one)
print(state.c)
state_t = tf.transpose(state.c)
print(state_t)
#state_c = tf.multiply(state.c, zero_one)

outputs, state = tf.nn.static_rnn(cell, sequences, initial_state = state)
outputs = tf.reshape(tf.concat(outputs, 1), [batch_size, subsequence_length, lstm_memory_size])


my_inputs = np.random.rand(batch_size, subsequence_length, lstm_memory_size)
custom_cell_state = np.random.rand(batch_size, lstm_memory_size)
#print(custom_cell_state)
#print(my_inputs)



def zero_some_states(action_array, states):
    """
    changes the states of a BasicLSTM-StateTuple to 0 of some of the samples in the batch

    :param action_array: 1 means no reset and 0 resets the cell-state
    :param states: tf.nn.rnn_cell.LSTMStateTuple(c = cell_state,
                                                 h = hidden_state)
    """
    for step, action in enumerate(action_array):
        if action == 1:
            pass
        elif action == 0:
            states.h[step] = 0
            states.c[step] = 0
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

    #zero_some_states([0, 1, 0], state_1)
    print('state1.h ',state_1.h)

    out_3, state_3 = session.run((outputs, state),
                                 feed_dict={inputs: out_1, cell_state: state_1.c, hidden_state: state_1.h})
    print("val3", np.mean(out_3))
    variables_names = [v.name for v in tf.global_variables()]
    #out_2, cell_s
    #print(hidden_state)
