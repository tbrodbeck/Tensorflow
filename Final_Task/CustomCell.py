import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops import init_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class CustomBasicLSTMCell(rnn.BasicLSTMCell):

    def __init__(self, num_units, forget_bias=1.0,
                 state_is_tuple=True, activation=None, reuse=None, name=None, custom_settings=False, custom_bias=None,
                 custom_weights=None):
        """Initialize a costum BasicLSTMCell with additional initializer for bias and weight

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          custom_settings: If true, custom_bias and custom_weights are used and have to be set
          custom_bias: initializier for bias
          custom_weights: initializer for weights
          When restoring from CudnnLSTM-trained checkpoints, must use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(CustomBasicLSTMCell, self).__init__(num_units, forget_bias, state_is_tuple, activation, reuse, name)

        self._custom_bias = custom_bias
        self._custom_weights = custom_weights
        self._custom_settings = custom_settings

    # custom getter
    @property
    def parameters(self):
        return (self._kernel, self._bias)


    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units

        # custom initializer possibility
        if self._custom_settings:
            self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME,
                                             shape=[input_depth + h_depth, 4 * self._num_units],
                                             initializer=self._custom_weights)
            self._bias = self.add_variable(_BIAS_VARIABLE_NAME,
                                           shape=[4 * self._num_units],
                                           initializer=self._custom_bias)
        else:
            self._kernel = self.add_variable(
                _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + h_depth, 4 * self._num_units])
            self._bias = self.add_variable(
                _BIAS_VARIABLE_NAME,
                shape=[4 * self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True
