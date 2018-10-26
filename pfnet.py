from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from transformer.spatial_transformer import transformer
from utils.network_layers import conv2_layer, locallyconn2_layer, dense_layer


class PFCell(tf.nn.rnn_cell.RNNCell):
    """
    PF-net for localization implemented with the RNN interface.
    Implements the particle set update, the observation and transition models.
    Cell inputs: observation, odometry
    Cell states: particle_states, particle_weights
    Cell outputs: particle_states, particle_weights
    """

    def __init__(self, global_maps, params, batch_size, num_particles):
        """
        :param global_maps: tensorflow op (batch, None, None, ch), global maps input. Since the map is fixed
        through the trajectory it can be input to the cell here, instead of part of the cell input.
        :param params: parsed arguments
        :param batch_size: int, minibatch size
        :param num_particles: number of particles
        """
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
        """
        Implements a particle update.
        :param inputs: observation (batch, 56, 56, ch), odometry (batch, 3).
        observation is the sensor reading at time t, odometry is the relative motion from time t to time t+1
        :param state: particle states (batch, K, 3), particle weights (batch, K).
        weights are assumed to be in log space and they can be unnormalized
        :param scope: not used, only kept for the interface. Ops will be created in the current scope.
        :return: outputs, state
        outputs: particle states and weights after the observation update, but before the transition update
        state: updated particle states and weights after both observation and transition updates
        """
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
        Implements a stochastic transition model for localization.
        :param particle_states: tf op (batch, K, 3), particle states before the update.
        :param odometry: tf op (batch, 3), odometry reading, relative motion in the robot coordinate frame
        :return: particle_states updated with the odometry and optionally transition noise
        """
        translation_std = self.params.transition_std[0] / self.params.map_pixel_in_meters  # in pixels
        rotation_std = self.params.transition_std[1]  # in radians

        with tf.name_scope('transition'):
            part_x, part_y, part_th = tf.unstack(particle_states, axis=-1, num=3)

            odometry = tf.expand_dims(odometry, axis=1)
            odom_x, odom_y, odom_th = tf.unstack(odometry, axis=-1, num=3)

            noise_th = tf.random_normal(part_th.get_shape(), mean=0.0, stddev=1.0) * rotation_std

            # add orientation noise before translation
            part_th += noise_th

            cos_th = tf.cos(part_th)
            sin_th = tf.sin(part_th)
            delta_x = cos_th * odom_x - sin_th * odom_y
            delta_y = sin_th * odom_x + cos_th * odom_y
            delta_th = odom_th

            delta_x += tf.random_normal(delta_x.get_shape(), mean=0.0, stddev=1.0) * translation_std
            delta_y += tf.random_normal(delta_y.get_shape(), mean=0.0, stddev=1.0) * translation_std

            return tf.stack([part_x+delta_x, part_y+delta_y, part_th+delta_th], axis=-1)

    def observation_model(self, global_maps, particle_states, observation):
        """
        Implements a discriminative observation model for localization.
        The model transforms the single global map to local maps for each particle, where a local map is a local
        view from the state defined by the particle.
        :param global_maps: tf op (batch, None, None, ch), global maps input.
        Assumes a scaling 0..2 where 0 is occupied, 2 is free space.
        :param particle_states: tf op (batch, K, 3), particle states before the update
        :param observation: tf op (batch, 56, 56, ch), image observation from a rgb, depth, or rgbd camera.
        :return: tf op (batch, K) particle likelihoods in the log space, unnormalized
        """

        # transform global maps to local maps
        local_maps = self.transform_maps(global_maps, particle_states, (28, 28))

        # rescale from 0..2 to -1..1. This is not actually necessary.
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

    @staticmethod
    def resample(particle_states, particle_weights, alpha):
        """
        Implements (soft)-resampling of particles.
        :param particle_states: tf op (batch, K, 3), particle states
        :param particle_weights: tf op (batch, K), unnormalized particle weights in log space
        :param alpha: float, trade-off parameter for soft-resampling. alpha == 1 corresponds to standard,
        hard-resampling. alpha == 0 corresponds to sampling particles uniformly, ignoring their weights.
        :return: particle_states, particle_weights
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

    @staticmethod
    def transform_maps(global_maps, particle_states, local_map_size):
        """
        Implements global to local map transformation
        :param global_maps: tf op (batch, None, None, ch) global map input
        :param particle_states: tf op (batch, K, 3) particle states that define local views for the transformation
        :param local_map_size: tuple, (height, widght), size of the output local maps
        :return: tf op (batch, K, local_map_size[0], local_map_size[1], ch). local maps, each shows a
          different transformation of the global map corresponding to the particle states
        """
        batch_size, num_particles = particle_states.get_shape().as_list()[:2]
        total_samples = batch_size * num_particles
        flat_states = tf.reshape(particle_states, [total_samples, 3])

        # define some helper variables
        input_shape = tf.shape(global_maps)
        global_height = tf.cast(input_shape[1], tf.float32)
        global_width = tf.cast(input_shape[2], tf.float32)
        height_inverse = 1.0 / global_height
        width_inverse = 1.0 / global_width
        # at tf1.6 matmul still does not support broadcasting, so we need full vectors
        zero = tf.constant(0, dtype=tf.float32, shape=(total_samples, ))
        one = tf.constant(1, dtype=tf.float32, shape=(total_samples, ))

        # the global map will be down-scaled by some factor
        window_scaler = 8.0

        # normalize orientations and precompute cos and sin functions
        theta = -flat_states[:, 2] - 0.5 * np.pi
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)

        # construct an affine transformation matrix step-by-step.
        # 1, translate the global map s.t. the center is at the particle state
        translate_x = (flat_states[:, 0] * width_inverse * 2.0) - 1.0
        translate_y = (flat_states[:, 1] * height_inverse * 2.0) - 1.0

        transm1 = tf.stack((one, zero, translate_x, zero, one, translate_y, zero, zero, one), axis=1)
        transm1 = tf.reshape(transm1, (total_samples, 3, 3))

        # 2, rotate map s.t. the orientation matches that of the particles
        rotm = tf.stack((costheta, sintheta, zero, -sintheta, costheta, zero, zero, zero, one), axis=1)
        rotm = tf.reshape(rotm, (total_samples, 3, 3))

        # 3, scale down the map
        scale_x = tf.fill((total_samples, ), float(local_map_size[1] * window_scaler) * width_inverse)
        scale_y = tf.fill((total_samples, ), float(local_map_size[0] * window_scaler) * height_inverse)

        scalem = tf.stack((scale_x, zero, zero, zero, scale_y, zero, zero, zero, one), axis=1)
        scalem = tf.reshape(scalem, (total_samples, 3, 3))

        # 4, translate the local map s.t. the particle defines the bottom mid-point instead of the center
        translate_y2 = tf.constant(-1.0, dtype=tf.float32, shape=(total_samples, ))

        transm2 = tf.stack((one, zero, zero, zero, one, translate_y2, zero, zero, one), axis=1)
        transm2 = tf.reshape(transm2, (total_samples, 3, 3))

        # chain the transformation matrices into a single one: translate + rotate + scale + translate
        transform_m = tf.matmul(tf.matmul(tf.matmul(transm1, rotm), scalem), transm2)

        # reshape to the format expected by the spatial transform network
        transform_m = tf.reshape(transform_m[:, :2], (batch_size, num_particles, 6))

        # do the image transformation using the spatial transform network
        # iterate over particle to avoid tiling large global maps
        output_list = []
        for i in range(num_particles):
            output_list.append(transformer(global_maps, transform_m[:,i], local_map_size))

        local_maps = tf.stack(output_list, axis=1)

        # set shape information that is lost in the spatial transform network
        local_maps = tf.reshape(local_maps, (batch_size, num_particles, local_map_size[0], local_map_size[1],
                                             global_maps.shape.as_list()[-1]))

        return local_maps

    @staticmethod
    def map_features(local_maps):
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

    @staticmethod
    def observation_features(observation):
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

    @staticmethod
    def joint_matrix_features(joint_matrix):
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

    @staticmethod
    def joint_vector_features(joint_vector):
        with tf.variable_scope("joint"):
            x = joint_vector
            x = dense_layer(1, activation=None, use_bias=True, name='fc1')(x)
        return x


class PFNet(object):
    """ Implements PF-net. Unrolls the PF-net RNN cell and defines losses and training ops."""
    def __init__(self, inputs, labels, params, is_training):
        """
        Calling this will create all tf ops for PF-net.
        :param inputs: list of tf ops, the inputs to PF-net. Assumed to have the following elements:
        global_maps, init_particle_states, observations, odometries, is_first_step
        :param labels: tf op, labels for training. Assumed to be the true states along the trajectory.
        :param params: parsed arguments
        :param is_training: bool, true for training.
        """
        self.params = params

        # define ops to be accessed conveniently from outside
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
        """
        Unroll the PF-net RNN cell and create loss ops and optionally, training ops
        """
        self.outputs = self.build_rnn(*inputs)

        self.build_loss_op(self.outputs[0], self.outputs[1], true_states=labels)

        if is_training:
            self.build_train_op()

    def save_state(self, sess):
        """
        Returns a list, the hidden state of PF-net, i.e. the particle states and particle weights.
        The output can be used with load_state to restore the current hidden state.
        """
        return sess.run(self.hidden_states)

    def load_state(self, sess, saved_state):
        """
        Overwrite the hidden state of PF-net to that of saved_state.
        """
        return sess.run(self.hidden_states,
                        feed_dict={self.hidden_states[i]: saved_state[i] for i in range(len(self.hidden_states))})

    def build_loss_op(self, particle_states, particle_weights, true_states):
        """
        Create tf ops for various losses. This should be called only once with is_training=True.
        """
        assert particle_weights.get_shape().ndims == 3

        lin_weights = tf.nn.softmax(particle_weights, dim=-1)

        true_coords = true_states[:, :, :2]
        mean_coords = tf.reduce_sum(tf.multiply(particle_states[:,:,:,:2], lin_weights[:,:,:,None]), axis=2)
        coord_diffs = mean_coords - true_coords

        # convert from pixel coordinates to meters
        coord_diffs *= self.params.map_pixel_in_meters

        # coordinate loss component: (x-x')^2 + (y-y')^2
        loss_coords = tf.reduce_sum(tf.square(coord_diffs), axis=2)

        true_orients = true_states[:, :, 2]
        orient_diffs = particle_states[:, :, :, 2] - true_orients[:,:,None]
        # normalize between -pi..+pi
        orient_diffs = tf.mod(orient_diffs + np.pi, 2*np.pi) - np.pi
        # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        loss_orient = tf.square(tf.reduce_sum(orient_diffs * lin_weights, axis=2))

        # combine translational and orientation losses
        loss_combined = loss_coords + 0.36 * loss_orient
        loss_pred = tf.reduce_mean(loss_combined, name='prediction_loss')

        # add L2 regularization loss
        loss_reg = tf.multiply(tf.losses.get_regularization_loss(), self.params.l2scale, name='l2')
        loss_total = tf.add_n([loss_pred, loss_reg], name="training_loss")

        self.all_distance2_op = loss_coords
        self.valid_loss_op = loss_pred
        self.train_loss_op = loss_total

        return loss_total

    def build_train_op(self):
        """ Create optimizer and train op. This should be called only once. """

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
        """
        Unroll the PF-net RNN cell through time. Input arguments are the inputs to PF-net. The time dependent
        fields are expected to be broken into fixed-length segments defined by params.bptt_steps
        """
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

        # choose state for the current trajectory segment
        state = tf.cond(is_first_step,
                        true_fn=lambda: (init_particle_states, init_particle_weights),
                        false_fn=lambda: tuple(self.hidden_states))

        with tf.variable_scope("rnn"):
            # hack to create variables on GPU
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

            # unroll real steps using the variables already created
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

        # define an op to update the hidden state, i.e. the particle states and particle weights.
        # this should be evaluated after every input
        with tf.control_dependencies([particle_states, particle_weights]):
            self.update_state_op = tf.group(
                *(self.hidden_states[i].assign(state[i]) for i in range(len(self.hidden_states))))

        return particle_states, particle_weights

