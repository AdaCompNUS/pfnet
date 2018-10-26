from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, pickle
import tensorflow as tf
import numpy as np
import cv2

from utils.tfrecordfeatures import *

from tensorpack import dataflow
from tensorpack.dataflow.base import RNGDataFlow, DataFlow, ProxyDataFlow

try:
    import ipdb as pdb
except Exception:
    import pdb


def py_proc_image(img, resize=None, display=False):
    nparr = np.fromstring(img, np.uint8)
    img = cv2.imdecode(nparr, -1)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


def raw_images_to_array(images):
    image_list = []
    for image_str in images:
        image = py_proc_image(image_str, (56, 56))
        image = scale_observation(np.atleast_3d(image.astype(np.float32)))
        image_list.append(image)

    return np.stack(image_list, axis=0)


def scale_observation(x):
    if x.ndim == 2 or x.shape[2] == 1:  # depth
        return x * (2.0 / 100.0) - 1.0
    else:  # rgb
        return x * (2.0/255.0) - 1.0


def tf_scale_sensor(x, mode):
    if mode == "both":
        return tf_scale_sensor(x, ("depth" if x.ndim == 2 or x.shape[2] == 1 else "rgb"))
    elif mode == "depth" or mode == "laser":
        return tf.cast(x, tf.float32)*0.05 - 1
    elif mode == "rgb":
        return tf.cast(x, tf.float32)*(2.0/255.0) - 1
    return None


def inverse_scale_sensor(x, mode):
    if mode == "both":
        return inverse_scale_sensor(x, ("depth" if x.ndim == 2 or x.shape[2] == 1 else "rgb"))
    elif mode == "depth" or mode == "laser":
        return (x + 1.0) * (100.0 / 2.0)
    elif mode == "rgb":
        return (x + 1.0) * (255.0 / 2.0)
    return None


def bounding_box(img):
    # helper function to get the bounding box of nonzeros in a 2D map. bounding box is inclusive
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


class BatchDataWithPad(dataflow.BatchData):
    """
    Stack datapoints into batches, where certain elements are padded to be the same size
    in a single batch.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False, padded_indices=(),
                 central_padded_indices=()):
        """
        """
        super(BatchDataWithPad, self).__init__(ds, batch_size, remainder, use_list)
        self.padded_indices = padded_indices
        self.central_padded_indices = central_padded_indices

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchDataWithPad._aggregate_batch(holder, self.use_list, self.padded_indices,
                                                        self.central_padded_indices)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchDataWithPad._aggregate_batch(holder, self.use_list, self.padded_indices,
                                                    self.central_padded_indices)

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False, padded_indices=(), central_padded_indices=()):
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
                    shapes = np.array([x[k].shape for x in data_holder], 'i')  # assumes ndim are the same for all
                    assert shapes.shape[1] == 3  # only supports 3D arrays for now, e.g. images (height, width, ch)
                    matching_shape = shapes.max(axis=0).tolist()
                    new_data = np.zeros([shapes.shape[0]] + matching_shape, dtype=tp)
                    for i in range(len(data_holder)):
                        shape = data_holder[i][k].shape
                        new_data[i, :shape[0], :shape[1], :shape[2]] = data_holder[i][k]
                    result.append(new_data)
                elif k in central_padded_indices:
                    shapes = np.array([x[k].shape for x in data_holder], 'i')  # assumes ndim are the same for all
                    assert shapes.shape[1] == 5
                    # only supports 5D arrays for now, e.g. N conv kernels (N, kern, kern, in, out)
                    # different kernels are assumed, N, in, out should be the same
                    matching_shape = shapes.max(axis=0).tolist()
                    new_data = np.zeros([shapes.shape[0]] + matching_shape, dtype=tp)
                    for i in range(len(data_holder)):
                        shape = data_holder[i][k].shape
                        assert shape[1] == shape[2] and shape[1] % 2 == 1
                        diff = (matching_shape[1] - shape[1])//2
                        new_data[i, :, diff:shape[1]+diff, diff:shape[2]+diff, :, :] = data_holder[i][k]
                    result.append(new_data)
                else:
                    # standard batching
                    result.append(
                        np.asarray([x[k] for x in data_holder], dtype=tp))
            except Exception as e:  # noqa
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
    def __init__(self, ds, timed_indices, trajlen, bptt_steps):
        super(BreakForBPTT, self).__init__(ds)
        self.timed_indiced = timed_indices
        self.bptt_steps = bptt_steps

        assert trajlen % bptt_steps == 0
        self.num_segments = trajlen // bptt_steps

    def size(self):
        return self.ds.size() * self.num_segments

    def get_data(self):
        for data in self.ds.get_data():
            for split_i in range(self.num_segments):
                new_data = []
                for i in range(len(data)):
                   if (i in self.timed_indiced):
                       new_data.append(data[i][:, split_i*self.bptt_steps:(split_i+1)*self.bptt_steps])
                   else:
                       new_data.append(data[i])

                new_data.append((split_i == 0))

                yield new_data


class House3DTrajData(RNGDataFlow):
    def __init__(self, files, mapmode, obsmode, trajlen, num_particles, init_particles_distr,
                 init_particles_cov, seed=None):
        self.files = files
        self.mapmode = mapmode
        self.obsmode = obsmode
        self.trajlen = trajlen
        self.num_particles = num_particles
        self.init_particles_distr = init_particles_distr
        self.init_particles_cov = init_particles_cov
        self.seed = seed

        # count total entries
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
        super(House3DTrajData, self).reset_state()
        if self.seed:
            np.random.seed(1)
        else:
            np.random.seed(self.rng.randint(0, 99999999))

    def get_data(self):
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
        floormap = np.atleast_3d(py_proc_image(wallmap_feature))
        # transpose and invert
        floormap = 255 - np.transpose(floormap, axes=[1, 0, 2])
        return floormap

    def process_door_map(self, doormap_feature):
        return self.process_wall_map(doormap_feature)

    def process_roomtype_map(self, roomtypemap_feature):
        # 9
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
        roomidmap = np.atleast_3d(py_proc_image(roomidmap_feature))
        # this is NOT transposed, unlike other maps
        return roomidmap

    @staticmethod
    def random_particles(state, distr, particles_cov, num_particles, roomidmap, seed=None):
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

        # restore random seed
        if seed is not None:
            np.random.set_state(random_state)

        return particles

    @staticmethod
    def get_sample_seed(seed, data_i):
        return (None if (seed is None or seed == 0) else ((data_i + 1) * 113 + seed))


def get_dataflow(files, params, is_training):

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
