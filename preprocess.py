from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import cv2

from tensorpack import dataflow
from tensorpack.dataflow.base import RNGDataFlow, ProxyDataFlow

try:
    import ipdb as pdb
except Exception:
    import pdb


def decode_image(img_str, resize=None):
    """
    Decode image from tfrecord data
    :param img_str: image encoded as a png in a string
    :param resize: tuple width two elements that defines the new size of the image. optional
    :return: image as a numpy array
    """
    nparr = np.fromstring(img_str, np.uint8)
    img_str = cv2.imdecode(nparr, -1)
    if resize is not None:
        img_str = cv2.resize(img_str, resize)
    return img_str


def raw_images_to_array(images):
    """
    Decode and normalize multiple images from tfrecord data
    :param images: list of images encoded as a png in a string
    :return: a numpy array of size (N, 56, 56, channels), normalized for training
    """
    image_list = []
    for image_str in images:
        image = decode_image(image_str, (56, 56))
        image = scale_observation(np.atleast_3d(image.astype(np.float32)))
        image_list.append(image)

    return np.stack(image_list, axis=0)


def scale_observation(x):
    """
    Normalizes observation input, either an rgb image or a depth image
    :param x: observation input as numpy array, either an rgb image or a depth image
    :return: numpy array, a normalized observation
    """
    if x.ndim == 2 or x.shape[2] == 1:  # depth
        return x * (2.0 / 100.0) - 1.0
    else:  # rgb
        return x * (2.0/255.0) - 1.0


def bounding_box(img):
    """
    Bounding box of non-zeros in an array (inclusive). Used with 2D maps
    :param img: numpy array
    :return: inclusive bounding box indices: top_row, bottom_row, leftmost_column, rightmost_column
    """
    # helper function to
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


class BatchDataWithPad(dataflow.BatchData):
    """
    Stacks datapoints into batches. Selected elements can be padded to the same size in each batch.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False, padded_indices=()):
        """
        :param ds: input dataflow. Same as BatchData
        :param batch_size: mini batch size. Same as BatchData
        :param remainder: if data is not enough to form a full batch, it makes a smaller batch when true.
        Same as BatchData.
        :param use_list: if True, components will contain a list of datapoints instead of creating a new numpy array.
        Same as BatchData.
        :param padded_indices: list of filed indices for which all elements will be padded with zeros to mach
        the largest in the batch. Each batch may produce a different size datapoint.
        """
        super(BatchDataWithPad, self).__init__(ds, batch_size, remainder, use_list)
        self.padded_indices = padded_indices

    def get_data(self):
        """
        Yields:  Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchDataWithPad._aggregate_batch(holder, self.use_list, self.padded_indices)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchDataWithPad._aggregate_batch(holder, self.use_list, self.padded_indices)

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False, padded_indices=()):
        """
        Re-implement the parent function with the option to pad selected fields to the largest in the batch.
        """
        assert not use_list  # cannot match shape if they must be treated as lists
        size = len(data_holder[0])
        result = []
        for k in range(size):
            dt = data_holder[0][k]
            if type(dt) in [int, bool]:
                tp = 'int32'
            elif type(dt) == float:
                tp = 'float32'
            else:
                try:
                    tp = dt.dtype
                except AttributeError:
                    raise TypeError("Unsupported type to batch: {}".format(type(dt)))
            try:
                if k in padded_indices:
                    # pad this field
                    shapes = np.array([x[k].shape for x in data_holder], 'i')  # assumes ndim are the same for all
                    assert shapes.shape[1] == 3  # only supports 3D arrays for now, e.g. images (height, width, ch)
                    matching_shape = shapes.max(axis=0).tolist()
                    new_data = np.zeros([shapes.shape[0]] + matching_shape, dtype=tp)
                    for i in range(len(data_holder)):
                        shape = data_holder[i][k].shape
                        new_data[i, :shape[0], :shape[1], :shape[2]] = data_holder[i][k]
                    result.append(new_data)
                else:
                    # no need to pad this field, simply create batch
                    result.append(np.asarray([x[k] for x in data_holder], dtype=tp))
            except Exception as e:
                # exception handling. same as in parent class
                pdb.set_trace()
                dataflow.logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                if isinstance(dt, np.ndarray):
                    s = dataflow.pprint.pformat([x[k].shape for x in data_holder])
                    dataflow.logger.error("Shape of all arrays to be batched: " + s)
                try:
                    # open an ipython shell if possible
                    import IPython as IP; IP.embed()    # noqa
                except ImportError:
                    pass
        return result


class BreakForBPTT(ProxyDataFlow):
    """
    Breaks long trajectories into multiple smaller segments for training with BPTT.
    Adds an extra field for indicating the first segment of a trajectory.
    """
    def __init__(self, ds, timed_indices, trajlen, bptt_steps):
        """
        :param ds: input dataflow
        :param timed_indices: field indices for which the second dimension corresponds to timestep along the trajectory
        :param trajlen: full length of trajectories
        :param bptt_steps: segment length, number of backprop steps for BPTT. Must be an integer divisor of trajlen
        """
        super(BreakForBPTT, self).__init__(ds)
        self.timed_indiced = timed_indices
        self.bptt_steps = bptt_steps

        assert trajlen % bptt_steps == 0
        self.num_segments = trajlen // bptt_steps

    def size(self):
        return self.ds.size() * self.num_segments

    def get_data(self):
        """
        Yields multiple datapoints per input datapoints corresponding segments of the trajectory.
        Adds an extra field for indicating the first segment of a trajectory.
        """

        for data in self.ds.get_data():
            for split_i in range(self.num_segments):
                new_data = []
                for i in range(len(data)):
                   if i in self.timed_indiced:
                       new_data.append(data[i][:, split_i*self.bptt_steps:(split_i+1)*self.bptt_steps])
                   else:
                       new_data.append(data[i])

                new_data.append((split_i == 0))

                yield new_data


class House3DTrajData(RNGDataFlow):
    """
    Process tfrecords data of House3D trajectories. Produces a dataflow with the following fields:
    true state, global map, initial particles, observations, odometries
    """

    def __init__(self, files, mapmode, obsmode, trajlen, num_particles, init_particles_distr, init_particles_cov,
                 seed=None):
        """
        :param files: list of data file names. assumed to be tfrecords files
        :param mapmode: string, map type. Possible values: wall / wall-door / wall-roomtype / wall-door-roomtype
        :param obsmode: string, observation type. Possible values: rgb / depth / rgb-depth. Vrf is not yet supported
        :param trajlen: int, length of trajectories
        :param num_particles: int, number of particles
        :param init_particles_distr: string, type of initial particle distribution.
        Possible values: tracking / one-room. Does not support two-rooms and all-rooms yet.
        :param init_particles_cov: numpy array of shape (3,3), coveriance matrix for the initial particles. Ignored
        when init_particles_distr != 'tracking'.
        :param seed: int or None. Random seed will be fixed if not None.
        """
        self.files = files
        self.mapmode = mapmode
        self.obsmode = obsmode
        self.trajlen = trajlen
        self.num_particles = num_particles
        self.init_particles_distr = init_particles_distr
        self.init_particles_cov = init_particles_cov
        self.seed = seed

        # count total number of entries
        count = 0
        for f in self.files:
            if not os.path.isfile(f):
                raise ValueError('Failed to find file: ' + f)
            record_iterator = tf.python_io.tf_record_iterator(f)
            for _ in record_iterator:
                count += 1
        self.count = count

    def size(self):
        return self.count

    def reset_state(self):
        """ Reset state. Fix numpy random seed if needed."""
        super(House3DTrajData, self).reset_state()
        if self.seed is not None:
            np.random.seed(1)
        else:
            np.random.seed(self.rng.randint(0, 99999999))

    def get_data(self):
        """
        Yields datapoints, all numpy arrays, with the following fields.

        true states: (trajlen, 3). Second dimension corresponds to x, y, theta coordinates.

        global map: (n, m, ch). shape is different for each map. number of channels depend on the mapmode setting

        initial particles: (num_particles, 3)

        observations: (trajlen, 56, 56, ch) number of channels depend on the obsmode setting

        odometries: (trajlen, 3) relative motion in the robot coordinate frame
        """
        for file in self.files:
            gen = tf.python_io.tf_record_iterator(file)
            for data_i, string_record in enumerate(gen):
                result = tf.train.Example.FromString(string_record)
                features = result.features.feature

                # process maps
                map_wall = self.process_wall_map(features['map_wall'].bytes_list.value[0])
                global_map_list = [map_wall]
                if 'door' in self.mapmode:
                    map_door = self.process_door_map(features['map_door'].bytes_list.value[0])
                    global_map_list.append(map_door)
                if 'roomtype' in self.mapmode:
                    map_roomtype = self.process_roomtype_map(features['map_roomtype'].bytes_list.value[0])
                    global_map_list.append(map_roomtype)
                if self.init_particles_distr == 'tracking':
                    map_roomid = None
                else:
                    map_roomid = self.process_roomid_map(features['map_roomid'].bytes_list.value[0])

                # input global map is a concatentation of semantic channels
                global_map = np.concatenate(global_map_list, axis=-1)

                # rescale to 0..2 range. this way zero padding will produce the equivalent of obstacles
                global_map = global_map.astype(np.float32) * (2.0 / 255.0)

                # process true states
                true_states = features['states'].bytes_list.value[0]
                true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

                # trajectory may be longer than what we use for training
                data_trajlen = true_states.shape[0]
                assert data_trajlen >= self.trajlen
                true_states = true_states[:self.trajlen]

                # process odometry
                odometry = features['odometry'].bytes_list.value[0]
                odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

                # process observations
                assert self.obsmode in ['rgb', 'depth', 'rgb-depth']  #TODO support for lidar
                if 'rgb' in self.obsmode:
                    rgb = raw_images_to_array(list(features['rgb'].bytes_list.value)[:self.trajlen])
                    observation = rgb
                if 'depth' in self.obsmode:
                    depth = raw_images_to_array(list(features['depth'].bytes_list.value)[:self.trajlen])
                    observation = depth
                if self.obsmode == 'rgb-depth':
                    observation = np.concatenate((rgb, depth), axis=-1)

                # generate particle states
                init_particles = self.random_particles(true_states[0], self.init_particles_distr,
                                                       self.init_particles_cov, self.num_particles,
                                                       roomidmap=map_roomid,
                                                       seed=self.get_sample_seed(self.seed, data_i), )

                yield (true_states, global_map, init_particles, observation, odometry)

    def process_wall_map(self, wallmap_feature):
        floormap = np.atleast_3d(decode_image(wallmap_feature))
        # transpose and invert
        floormap = 255 - np.transpose(floormap, axes=[1, 0, 2])
        return floormap

    def process_door_map(self, doormap_feature):
        return self.process_wall_map(doormap_feature)

    def process_roomtype_map(self, roomtypemap_feature):
        binary_map = np.fromstring(roomtypemap_feature, np.uint8)
        binary_map = cv2.imdecode(binary_map, 2)  # 16-bit image
        assert binary_map.dtype == np.uint16 and binary_map.ndim == 2
        # binary encoding from bit 0 .. 9

        room_map = np.zeros((binary_map.shape[0], binary_map.shape[1], 9), dtype=np.uint8)
        for i in range(9):
            room_map[:,:,i] = np.array((np.bitwise_and(binary_map, (1 << i)) > 0), dtype=np.uint8)
        room_map *= 255

        # transpose and invert
        room_map = np.transpose(room_map, axes=[1, 0, 2])
        return room_map

    def process_roomid_map(self, roomidmap_feature):
        # this is not transposed, unlike other maps
        roomidmap = np.atleast_3d(decode_image(roomidmap_feature))
        return roomidmap

    @staticmethod
    def random_particles(state, distr, particles_cov, num_particles, roomidmap, seed=None):
        """
        Generate a random set of particles
        :param state: true state, numpy array of x,y,theta coordinates
        :param distr: string, type of distribution. Possible values: tracking / one-room.
        For 'tracking' the distribution is a Gaussian centered near the true state.
        For 'one-room' the distribution is uniform over states in the room defined by the true state.
        :param particles_cov: numpy array of shape (3,3), defines the covariance matrix if distr == 'tracking'
        :param num_particles: number of particles
        :param roomidmap: numpy array, map of room ids. Values define a unique room id for each pixel of the map.
        :param seed: int or None. If not None, the random seed will be fixed for generating the particle.
        The random state is restored to its original value.
        :return: numpy array of particles (num_particles, 3)
        """
        assert distr in ["tracking", "one-room"]  #TODO add support for two-room and all-room

        particles = np.zeros((num_particles, 3), np.float32)

        if distr == "tracking":
            # fix seed
            if seed is not None:
                random_state = np.random.get_state()
                np.random.seed(seed)

            # sample offset from the Gaussian
            center = np.random.multivariate_normal(mean=state, cov=particles_cov)

            # restore random seed
            if seed is not None:
                np.random.set_state(random_state)

            # sample particles from the Gaussian, centered around the offset
            particles = np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles)

        elif distr == "one-room":
            # mask the room the initial state is in
            masked_map = (roomidmap == roomidmap[int(np.rint(state[0])), int(np.rint(state[1]))])

            # get bounding box for more efficient sampling
            rmin, rmax, cmin, cmax = bounding_box(masked_map)

            # rejection sampling inside bounding box
            sample_i = 0
            while sample_i < num_particles:
                particle = np.random.uniform(low=(rmin, cmin, 0.0), high=(rmax, cmax, 2.0*np.pi), size=(3, ),)
                # reject if mask is zero
                if not masked_map[int(np.rint(particle[0])), int(np.rint(particle[1]))]:
                    continue
                particles[sample_i] = particle
                sample_i += 1
        else:
            raise ValueError

        return particles

    @staticmethod
    def get_sample_seed(seed, data_i):
        """
        Defines a random seed for each datapoint in a deterministic manner.
        :param seed: int or None, defining a random seed
        :param data_i: int, the index of the current data point
        :return: None if seed is None, otherwise an int, a fixed function of both seed and data_i inputs.
        """
        return (None if (seed is None or seed == 0) else ((data_i + 1) * 113 + seed))


def get_dataflow(files, params, is_training):
    """
    Build a tensorflow Dataset from appropriate tfrecords files.
    :param files: list a file paths corresponding to appropriate tfrecords data
    :param params: parsed arguments
    :param is_training: bool, true for training.
    :return: (nextdata, num_samples).
    nextdata: list of tensorflow ops that produce the next input with the following elements:
    true_states, global_map, init_particles, observations, odometries, is_first_step.
    See House3DTrajData.get_data for definitions.
    num_samples: number of samples that make an epoch
    """

    mapmode = params.mapmode
    obsmode = params.obsmode
    batchsize = params.batchsize
    num_particles = params.num_particles
    trajlen = params.trajlen
    bptt_steps = params.bptt_steps

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std[0] = particle_std[0] / params.map_pixel_in_meters  # convert meters to pixels
    particle_std2 = np.square(particle_std)  # variance
    init_particles_cov = np.diag(particle_std2[(0, 0, 1),])

    df = House3DTrajData(files, mapmode, obsmode, trajlen, num_particles,
                         params.init_particles_distr, init_particles_cov,
                         seed=(params.seed if params.seed is not None and params.seed > 0
                               else (params.validseed if not is_training else None)))
    # data: true_states, global_map, init_particles, observation, odometry

    # make it a multiple of batchsize
    df = dataflow.FixedSizeData(df, size=(df.size() // batchsize) * batchsize, keep_state=False)

    # shuffle
    if is_training:
        df = dataflow.LocallyShuffleData(df, 100 * batchsize)

    # repeat data for the number of epochs
    df = dataflow.RepeatedData(df, params.epochs)

    # batch
    df = BatchDataWithPad(df, batchsize, padded_indices=(1,))

    # break trajectory into multiple segments for BPPT training. Augment df with is_first_step indicator
    df = BreakForBPTT(df, timed_indices=(0, 3, 4), trajlen=trajlen, bptt_steps=bptt_steps)
    # data: true_states, global_map, init_particles, observation, odometry, is_first_step

    num_samples = df.size() // params.epochs

    df.reset_state()

    # # test dataflow
    # df = dataflow.TestDataSpeed(dataflow.PrintData(df), 100)
    # df.start()

    obs_ch = {'rgb': 3, 'depth': 1, 'rgb-depth': 4}
    map_ch = {'wall': 1, 'wall-door': 2, 'wall-roomtype': 10, 'wall-door-roomtype': 11}
    types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool]
    sizes = [(batchsize, bptt_steps, 3),
             (batchsize, None, None, map_ch[mapmode]),
             (batchsize, num_particles, 3),
             (batchsize, bptt_steps, 56, 56, obs_ch[obsmode]),
             (batchsize, bptt_steps, 3),
             (), ]

    # turn it into a tf dataset
    def tuplegen():
        for dp in df.get_data():
            yield tuple(dp)

    dataset = tf.data.Dataset.from_generator(tuplegen, tuple(types), tuple(sizes))
    iterator = dataset.make_one_shot_iterator()
    nextdata = iterator.get_next()

    return nextdata, num_samples
