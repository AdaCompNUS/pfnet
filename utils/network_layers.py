import tensorflow as tf
import numpy as np

_L2_SCALE = 1.0  # scale when adding to loss instead of a global scaler


# Helper functions for constructing layers
def dense_layer(units, activation=None, use_bias=False, name=None):

    fn = lambda x: tf.layers.dense(
        x, units, activation=convert_activation_string(activation), use_bias=use_bias, name=name,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_L2_SCALE))
    return fn


def conv2_layer(filters, kernel_size, activation=None, padding='same', strides=(1,1), dilation_rate=(1,1),
                data_format='channels_last', use_bias=False, name=None, layer_i=0, name2=None):
    if name is None:
        name = "l%d_conv%d" % (layer_i, np.max(kernel_size))
        if np.max(dilation_rate) > 1:
            name += "_d%d" % np.max(dilation_rate)
        if name2 is not None:
            name += "_"+name2
    fn = lambda x: tf.layers.conv2d(
        x, filters, kernel_size, activation=convert_activation_string(activation),
        padding=padding, strides=strides, dilation_rate = dilation_rate, data_format=data_format,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_L2_SCALE),
        kernel_initializer=tf.variance_scaling_initializer(),
        use_bias=use_bias, name=name)
    return fn


def locallyconn2_layer(filters, kernel_size, activation=None, padding='same', strides=(1,1), dilation_rate=(1,1),
                data_format='channels_last', use_bias=False, name=None, layer_i=0, name2=None):
    assert dilation_rate == (1, 1)  # keras layer doesnt have this input. maybe different name?
    if name is None:
        name = "l%d_conv%d"%(layer_i, np.max(kernel_size))
        if np.max(dilation_rate) > 1:
            name += "_d%d"%np.max(dilation_rate)
        if name2 is not None:
            name += "_"+name2
    fn = tf.keras.layers.LocallyConnected2D(
        filters, kernel_size, activation=convert_activation_string(activation),
        padding=padding, strides=strides, data_format=data_format,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=_L2_SCALE),
        kernel_initializer=tf.variance_scaling_initializer(),
        use_bias=use_bias, name=name)
    return fn


def convert_activation_string(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation == 'tanh':
            activation = tf.nn.tanh
        else:
            assert False
    return activation
