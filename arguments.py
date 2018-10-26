import configargparse
import numpy as np


def parse_args(args=None):
    """
    Parse command line arguments
    :param args: command line arguments or None (default)
    :return: dictionary of parameters
    """

    p = configargparse.ArgParser(default_config_files=[])

    p.add('-c', '--config', required=True, is_config_file=True,
          help='Config file. use ./config/train.conf for training')

    p.add('--trainfiles', nargs='*', help='Data file(s) for training (tfrecord).')
    p.add('--testfiles', nargs='*', help='Data file(s) for validation or evaluation (tfrecord).')

    # input configuration
    p.add('--obsmode', type=str, default='rgb',
          help='Observation input type. Possible values: rgb / depth / rgb-depth / vrf.')
    p.add('--mapmode', type=str, default='wall',
          help='Map input type with different (semantic) channels. ' +
               'Possible values: wall / wall-door / wall-roomtype / wall-door-roomtype')
    p.add('--map_pixel_in_meters', type=float, default=0.02,
          help='The width (and height) of a pixel of the map in meters. Defaults to 0.02 for House3D data.')

    p.add('--init_particles_distr', type=str, default='tracking',
          help='Distribution of initial particles. Possible values: tracking / one-room / two-rooms / all-rooms')
    p.add('--init_particles_std', nargs='*', default=["0.3", "0.523599"],  # tracking setting, 30cm, 30deg
          help='Standard deviations for generated initial particles. Only applies to the tracking setting.' +
               'Expects two float values: translation std (meters), rotation std (radians)')
    p.add('--trajlen', type=int, default=24,
          help='Length of trajectories. Assumes lower or equal to the trajectory length in the input data.')

    # PF-net configuration
    p.add('--num_particles', type=int, default=30, help='Number of particles in PF-net.')
    p.add('--resample', type=str, default='false',
          help='Resample particles in PF-net. Possible values: true / false.')
    p.add('--alpha_resample_ratio', type=float, default=1.0,
          help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. '
               'Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
    p.add('--transition_std', nargs='*', default=["0.0", "0.0"],
                help='Standard deviations for transition model. Expects two float values: ' +
                     'translation std (meters), rotatation std (radians). Defaults to zeros.')

    # training configuration
    p.add('--batchsize', type=int, default=24, help='Minibatch size for training. Must be 1 for evaluation.')
    p.add('--bptt_steps', type=int, default=4,
          help='Number of backpropagation steps for training with backpropagation through time (BPTT). '
               'Assumed to be an integer divisor of the trajectory length (--trajlen).')
    p.add('--learningrate', type=float, default=0.0025, help='Initial learning rate for training.')
    p.add('--l2scale', type=float, default=4e-6, help='Scaling term for the L2 regularization loss.')
    p.add('--epochs', metavar='epochs', type=int, default=1, help='Number of epochs for training.')
    p.add('--decaystep', type=int, default=4, help='Decay the learning rate after every N epochs.')
    p.add('--decayrate', type=float, help='Rate of decaying the learning rate.')

    p.add('--load', type=str, default="", help='Load a previously trained model from a checkpoint file.')
    p.add('--logpath', type=str, default='',
          help='Specify path for logs. Makes a new directory under ./log/ if empty (default).')
    p.add('--seed', type=int, help='Fix the random seed of numpy and tensorflow if set to larger than zero.')
    p.add('--validseed', type=int,
          help='Fix the random seed for validation if set to larger than zero. ' +
               'Useful to evaluate with a fixed set of initial particles, which reduces the validation error variance.')
    p.add('--gpu', type=int, default=0, help='Select a gpu on a multi-gpu machine. Defaults to zero.')

    params = p.parse_args(args=args)

    # fix numpy seed if needed
    if params.seed is not None and params.seed >= 0:
        np.random.seed(params.seed)

    # convert multi-input fileds to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    # convert boolean fields
    if params.resample not in ['false', 'true']:
        print ("The value of resample must be either 'false' or 'true'")
        raise ValueError
    params.resample = (params.resample == 'true')

    return params
