import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import regularizers


def model_arg_scope(weight_decay=0.0005, weights_initializer=initializers.xavier_initializer(),
                    biases_initializer=init_ops.zeros_initializer()):
  with arg_scope(
      [layers.fully_connected, layers.conv2d],
      weights_initializer=weights_initializer,
      weights_regularizer=regularizers.l2_regularizer(weight_decay),
      biases_initializer=biases_initializer) as arg_sc:
    return arg_sc


def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf


def getActivationFunctionOp(act_name="relu"):
  if type(act_name) != str and type(act_name) != unicode:
    return act_name
  if act_name.lower() == 'relu':
    return tf.nn.relu
  elif act_name.lower() == 'tanh':
    return tf.nn.tanh
  elif act_name.lower() == 'lrelu':
    return lambda x: tf.nn.leaky_relu(x, alpha=0.01)
  elif act_name.lower() == 'llrelu':
    return lambda x: tf.nn.leaky_relu(x, alpha=0.1)
  elif act_name.lower() == 'gelu':
    return lambda x: gelu(x)
  else:
    return tf.nn.relu
