from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from transformer.spatial_transformer import transformer
from utils.network_layers import conv2_layer, locallyconn2_layer, dense_layer


class PFCell(tf.nn.rnn_cell.RNNCell):
    """RNN cell implementing PF-net for localization"""
    def __init__(self, global_maps, params, batch_size, num_particles):
        super(PFCell, self).__init__()
        self.global_maps = global_maps
        self.params = params
        self.batch_size = batch_size
        self.num_particles = num_particles

        self.states_shape = (batch_size, num_particles, 3)
        self.weights_shape = (batch_size, num_particles, )

    @property
    def state_size(self):
        return (tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]))

    @property
    def output_size(self):
        return (tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]))

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(tf.get_variable_scope()):
            particle_states, particle_weights = state
            observation, odometry = inputs

            # observation update
            lik = self.observation_model(self.global_maps, particle_states, observation)
            particle_weights += lik  # unnormalized

            # resample
            if self.params.resample:
                particle_states, particle_weights = self.resample(
                    particle_states, particle_weights, alpha=self.params.alpha_resample_ratio)

            # construct output before motion update
            outputs = particle_states, particle_weights

            # motion update. this will only affect the particle state input at the next step
            particle_states = self.transition_model(particle_states, odometry)

            # construct new state
            state = particle_states, particle_weights

        return outputs, state

    def transition_model(self, particle_states, odometry):
        """
        """
        with tf.name_scope('transition'):
            part_x, part_y, part_th = tf.unstack(particle_states, axis=-1, num=3)

            odometry = tf.expand_dims(odometry, axis=1)
            odom_x, odom_y, odom_th = tf.unstack(odometry, axis=-1, num=3)

            noise_th = tf.random_normal(part_th.get_shape(), mean=0.0, stddev=self.params.transition_std[1])

            # add orientation noise both before and after translation
            part_th += noise_th

            cos_th = tf.cos(part_th)
            sin_th = tf.sin(part_th)
            delta_x = cos_th * odom_x - sin_th * odom_y
            delta_y = sin_th * odom_x + cos_th * odom_y
            delta_th = odom_th

            delta_x += tf.random_normal(delta_x.get_shape(), mean=0.0, stddev=self.params.transition_std[0])
            delta_y += tf.random_normal(delta_y.get_shape(), mean=0.0, stddev=self.params.transition_std[0])
            delta_th += noise_th

            return tf.stack([part_x+delta_x, part_y+delta_y, part_th+delta_th], axis=-1)

    def observation_model(self, global_maps, particle_states, observation):
        """
        :param global_maps:
        :param particle_states:
        :param observations:
        :return:
        """

        # transform global maps to local maps
        local_maps = self.transform_maps(global_maps, particle_states, (28, 28))

        # rescale from 0..2 to -1..1
        local_maps = -(local_maps - 1)
        # flatten batch and particle dimensions
        local_maps = tf.reshape(local_maps,
                                [self.batch_size * self.num_particles] + local_maps.shape.as_list()[2:])

        # get features from the map
        map_features = self.map_features(local_maps)

        # get features from the observation
        obs_features = self.observation_features(observation)

        # tile observation features
        obs_features = tf.tile(tf.expand_dims(obs_features, axis=1), [1, self.num_particles, 1, 1, 1])
        obs_features = tf.reshape(obs_features,
                                  [self.batch_size * self.num_particles] + obs_features.shape.as_list()[2:])

        # sanity check
        assert obs_features.shape.as_list()[:-1] == map_features.shape.as_list()[:-1]

        # merge features and process further
        joint_features = tf.concat([map_features, obs_features], axis=-1)
        joint_features = self.joint_matrix_features(joint_features)

        # reshape to a vector and process further
        joint_features = tf.reshape(joint_features, (self.batch_size * self.num_particles, -1))
        lik = self.joint_vector_features(joint_features)
        lik = tf.reshape(lik, [self.batch_size, self.num_particles])

        return lik

    def resample(self, particle_states, particle_weights, alpha):
        """
        """
        with tf.name_scope('resample'):
            assert 0.0 < alpha <= 1.0
            batch_size, num_particles = particle_states.get_shape().as_list()[:2]

            # normalize
            particle_weights = particle_weights - tf.reduce_logsumexp(particle_weights, axis=-1, keep_dims=True)

            uniform_weights = tf.constant(-np.log(num_particles), shape=(batch_size, num_particles), dtype=tf.float32)

            # build sampling distribution, q(s), and update particle weights
            if alpha < 1.0:
                # soft resampling
                q_weights = tf.stack([particle_weights + np.log(alpha), uniform_weights + np.log(1.0-alpha)], axis=-1)
                q_weights = tf.reduce_logsumexp(q_weights, axis=-1, keep_dims=False)
                q_weights = q_weights - tf.reduce_logsumexp(q_weights, axis=-1, keep_dims=True)  # normalized

                particle_weights = particle_weights - q_weights  # this is unnormalized
            else:
                # hard resampling. this will produce zero gradients
                q_weights = particle_weights
                particle_weights = uniform_weights

            # sample particle indices according to q(s)
            indices = tf.cast(tf.multinomial(q_weights, num_particles), tf.int32)  # shape: (batch_size, num_particles)

            # index into particles
            helper = tf.range(0, batch_size*num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
            indices = indices + tf.expand_dims(helper, axis=1)

            particle_states = tf.reshape(particle_states, (batch_size * num_particles, 3))
            particle_states = tf.gather(particle_states, indices=indices, axis=0)  # (batch_size, num_particles, 3)

            particle_weights = tf.reshape(particle_weights, (batch_size * num_particles, ))
            particle_weights = tf.gather(particle_weights, indices=indices, axis=0)  # (batch_size, num_particles,)

            return particle_states, particle_weights

    def transform_maps(self, global_maps, particle_states, local_map_size):
        batch_size, num_particles = particle_states.get_shape().as_list()[:2]
        total_samples = batch_size * num_particles
        flat_states = tf.reshape(particle_states, [total_samples, 3])

        input_shape = tf.shape(global_maps)
        global_height = tf.cast(input_shape[1], tf.float32)
        global_width = tf.cast(input_shape[2], tf.float32)
        height_inverse = 1.0 / global_height
        width_inverse = 1.0 / global_width

        window_scaler = 8.0
        index_scaler = 1.0

        theta = -flat_states[:, 2] - 0.5 * np.pi
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)

        translate_x = (flat_states[:, 0] * width_inverse * 2.0 * index_scaler) - 1.0
        translate_y = (flat_states[:, 1] * height_inverse * 2.0 * index_scaler) - 1.0

        translate_y2 = tf.constant(-1.0, dtype=tf.float32, shape=(total_samples, ))

        scale_x = tf.fill((total_samples, ), float(local_map_size[1] * window_scaler) * width_inverse)
        scale_y = tf.fill((total_samples, ), float(local_map_size[0] * window_scaler) * height_inverse)

        # at tf1.6 matmul still does not support broadcasting, so we need full vectors
        zero = tf.constant(0, dtype=tf.float32, shape=(total_samples, ))
        one = tf.constant(1, dtype=tf.float32, shape=(total_samples, ))

        transm1 = tf.stack((one, zero, translate_x, zero, one, translate_y, zero, zero, one), axis=1)
        transm1 = tf.reshape(transm1, (total_samples, 3, 3))

        rotm = tf.stack((costheta, sintheta, zero, -sintheta, costheta, zero, zero, zero, one), axis=1)
        rotm = tf.reshape(rotm, (total_samples, 3, 3))

        scalem = tf.stack((scale_x, zero, zero, zero, scale_y, zero, zero, zero, one), axis=1)
        scalem = tf.reshape(scalem, (total_samples, 3, 3))

        transm2 = tf.stack((one, zero, zero, zero, one, translate_y2, zero, zero, one), axis=1)
        transm2 = tf.reshape(transm2, (total_samples, 3, 3))

        # translate, then rotate, then scale, then translate again
        transform_m = tf.matmul(tf.matmul(tf.matmul(transm1, rotm), scalem), transm2)
        transform_m = tf.reshape(transform_m[:, :2], (batch_size, num_particles, 6))

        # do transform separately for each particle to avoid tiling large global maps
        output_list = []
        for i in range(num_particles):
            output_list.append(transformer(global_maps, transform_m[:,i], local_map_size))

        local_maps = tf.stack(output_list, axis=1)
        # set shape
        local_maps = tf.reshape(local_maps, (batch_size, num_particles, local_map_size[0], local_map_size[1],
                                             global_maps.shape.as_list()[-1]))

        return local_maps

    def map_features(self, local_maps):
        assert local_maps.get_shape().as_list()[1:3] == [28, 28]
        data_format = 'channels_last'

        with tf.variable_scope("map"):
            x = local_maps
            layer_i = 1
            convs = [
                conv2_layer(
                    24, (3, 3), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    16, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    8, (7, 7), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    8, (7, 7), activation=None, padding='same', data_format=data_format,
                    dilation_rate=(2, 2), use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    8, (7, 7), activation=None, padding='same', data_format=data_format,
                    dilation_rate=(3, 3), use_bias=True, layer_i=layer_i)(x),
            ]
            x = tf.concat(convs, axis=-1)
            x = tf.contrib.layers.layer_norm(x, activation_fn=tf.nn.relu)
            assert x.get_shape().as_list()[1:4] == [28, 28, 64]
            # (28x28x64)

            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="same")

            layer_i+=1
            convs = [
                conv2_layer(
                    4, (3, 3), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    4, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                ]
            x = tf.concat(convs, axis=-1)
            x = tf.contrib.layers.layer_norm(x, activation_fn=tf.nn.relu)

        return x  # (14x14x8)

    def observation_features(self, observation):
        data_format = 'channels_last'
        with tf.variable_scope("observation"):
            x = observation
            layer_i = 1
            convs = [
                conv2_layer(
                    128, (3, 3), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    128, (5, 5), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    64, (5, 5), activation=None, padding='same', data_format=data_format,
                    dilation_rate=(2, 2), use_bias=True, layer_i=layer_i)(x),
                conv2_layer(
                    64, (5, 5), activation=None, padding='same', data_format=data_format,
                    dilation_rate=(4, 4), use_bias=True, layer_i=layer_i)(x),
            ]
            x = tf.concat(convs, axis=-1)
            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="same")
            x = tf.contrib.layers.layer_norm(x, activation_fn=tf.nn.relu)

            assert x.get_shape().as_list()[1:4] == [28, 28, 384]

            layer_i += 1
            x = conv2_layer(
                    16, (3, 3), activation=None, padding='same', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x)

            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="same")
            x = tf.contrib.layers.layer_norm(x, activation_fn=tf.nn.relu)
            assert x.get_shape().as_list()[1:4] == [14, 14, 16]

        return x  # (14,14,16)

    def joint_matrix_features(self, joint_matrix):
        assert joint_matrix.get_shape().as_list()[1:4] == [14, 14, 24]
        data_format = 'channels_last'

        with tf.variable_scope("joint"):
            x = joint_matrix
            layer_i = 1

            # pad manually to match different kernel sizes
            x_pad1 = tf.pad(x, paddings=tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]]))
            convs = [
                locallyconn2_layer(
                    8, (3, 3), activation='relu', padding='valid', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x),
                locallyconn2_layer(
                    8, (5, 5), activation='relu', padding='valid', data_format=data_format,
                    use_bias=True, layer_i=layer_i)(x_pad1),
            ]
            x = tf.concat(convs, axis=-1)

            x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2), padding="valid")
            assert x.get_shape().as_list()[1:4] == [5, 5, 16]

        return x  # (5, 5, 16)

    def joint_vector_features(self, joint_vector):
        with tf.variable_scope("joint"):
            x = joint_vector
            x = dense_layer(1, activation=None, use_bias=True, name='fc1')(x)
        return x


class PFNet(object):
    def __init__(self, inputs, labels, params, is_training):
        """
        :param params: dotdict with various parameters
        """
        self.params = params

        # define ops to be reached from outside
        self.outputs = []
        self.hidden_states = []

        self.train_loss_op = None
        self.valid_loss_op = None
        self.all_distance2_op = None

        self.global_step_op = None
        self.learning_rate_op = None
        self.train_op = None
        self.update_state_op = tf.constant(0)

        # build the network. this will generate the ops defined above
        self.build(inputs, labels, is_training)

    def build(self, inputs, labels, is_training):
        self.outputs = self.build_rnn(*inputs)

        self.build_loss_op(self.outputs[0], self.outputs[1], true_states=labels)

        if is_training:
            self.build_train_op()

    def save_state(self, sess):
        return sess.run(self.hidden_states)

    def load_state(self, sess, saved_state):
        return sess.run(self.hidden_states,
                        feed_dict={self.hidden_states[i]: saved_state[i] for i in range(len(self.hidden_states))})

    def build_loss_op(self, particle_states, particle_weights, true_states):
        # this should be called only once with is_training=True
        assert particle_weights.get_shape().ndims == 3

        lin_weights = tf.nn.softmax(particle_weights, dim=-1)

        true_coords = true_states[:, :, :2]
        mean_coords = tf.reduce_sum(tf.multiply(particle_states[:,:,:,:2], lin_weights[:,:,:,None]), axis=2)
        coord_diffs = mean_coords - true_coords

        # convert from pixel coordinates to meters
        coord_diffs *= 0.02

        # coordinate loss component: (x-x')^2 + (y-y')^2
        loss_coords = tf.reduce_sum(tf.square(coord_diffs), axis=2)

        true_orients = true_states[:, :, 2]
        orient_diffs = particle_states[:, :, :, 2] - true_orients[:,:,None]
        # normalize between -pi..+pi
        orient_diffs = tf.mod(orient_diffs + np.pi, 2*np.pi) - np.pi
        # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        loss_orient = tf.square(tf.reduce_sum(orient_diffs * lin_weights, axis=2))

        loss_combined = loss_coords + 0.36 * loss_orient

        loss_pred = tf.reduce_mean(loss_combined, name='prediction_loss')
        loss_reg = tf.multiply(tf.losses.get_regularization_loss(), self.params.l2scale, name='l2')

        loss_total = tf.add_n([loss_pred, loss_reg], name="training_loss")

        self.all_distance2_op = loss_coords
        self.valid_loss_op = loss_pred
        self.train_loss_op = loss_total

        return loss_total

    def build_train_op(self):
        # make sure this is only called once
        assert self.train_op is None and self.global_step_op is None and self.learning_rate_op is None

        # global step and learning rate
        with tf.device("/cpu:0"):
            self.global_step_op = tf.get_variable(
                initializer=tf.constant_initializer(0.0), shape=(), trainable=False, name='global_step',)
            self.learning_rate_op = tf.train.exponential_decay(
                self.params.learningrate, self.global_step_op, decay_steps=1, decay_rate=self.params.decayrate,
                staircase=True, name="learning_rate")

        # create gradient descent optimizer with the given learning rate.
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate_op, decay=0.9)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = optimizer.minimize(self.train_loss_op, global_step=None, var_list=tf.trainable_variables())

        return self.train_op

    def build_rnn(self, global_maps, init_particle_states, observations, odometries, is_first_step):
        batch_size, trajlen = observations.shape.as_list()[:2]
        num_particles = init_particle_states.shape.as_list()[1]
        global_map_ch = global_maps.shape.as_list()[-1]

        init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                            shape=(batch_size, num_particles), dtype=tf.float32)

        # create hidden state variable
        assert len(self.hidden_states) == 0  # no hidden state should be set before
        self.hidden_states = [
            tf.get_variable("particle_states", shape=init_particle_states.get_shape(),
                            dtype=init_particle_states.dtype, initializer=tf.constant_initializer(0), trainable=False),
            tf.get_variable("particle_weights", shape=init_particle_weights.get_shape(),
                            dtype=init_particle_weights.dtype, initializer=tf.constant_initializer(0), trainable=False),
            ]

        # choose state for this BPTT segment
        state = tf.cond(is_first_step,
                        true_fn=lambda: (init_particle_states, init_particle_weights),
                        false_fn=lambda: tuple(self.hidden_states))

        # hack to create variables on GPU
        with tf.variable_scope("rnn"):
            dummy_cell_func = PFCell(
                global_maps=tf.zeros((1, 1, 1, global_map_ch), dtype=global_maps.dtype),
                params=self.params, batch_size=1, num_particles=1)

            dummy_cell_func(
                  (tf.zeros([1]+observations.get_shape().as_list()[2:], dtype=observations.dtype),  # observation
                   tf.zeros([1, 3], dtype=odometries.dtype)),  # odometry
                  (tf.zeros([1, 1, 3], dtype=init_particle_states.dtype),  # particle_states
                   tf.zeros([1, 1], dtype=init_particle_weights.dtype)))  # particle_weights
            # variables are now created. set reuse
            tf.get_variable_scope().reuse_variables()

            cell_func = PFCell(global_maps=global_maps, params=self.params, batch_size=batch_size,
                                    num_particles=num_particles)

            outputs, state = tf.nn.dynamic_rnn(cell=cell_func,
                                               inputs=(observations, odometries),
                                               initial_state=state,
                                               swap_memory=True,
                                               time_major=False,
                                               parallel_iterations=1,
                                               scope=tf.get_variable_scope())

        particle_states, particle_weights = outputs

        with tf.control_dependencies([particle_states, particle_weights]):
            self.update_state_op = tf.group(
                *(self.hidden_states[i].assign(state[i]) for i in range(len(self.hidden_states))))

        return particle_states, particle_weights

