import configargparse
import numpy as np


def parse_args(args=None, return_parser=False):
    p = configargparse.ArgParser(default_config_files=[])

    p.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    p.add('--trainfiles', nargs='*', help='Directory for trainin set. (train)')
    p.add('--testfiles', nargs='*', help='Directory for test set. (test)')

    p.add('--mapmode', type=str, default='wall',
                    help='Map input mode, floor/door/type/doortype')
    p.add('--obsmode', type=str, default='rgb')
    p.add('--init_particles_distr', type=str, default='tracking')
    p.add('--l2scale', type=float,
                    help='Scale for L2 regularization term')

    p.add('--load', type=str, default="",
                    help='Load model weights from checkpoint')
    p.add('--epochs', metavar='epochs', type=int,
                    help='Max number of steps')
    p.add('--batchsize', type=int,
                    help='Minibatch size')
    p.add('--trajlen', type=int,
                    help='Length of trajectories. Same for each sample')
    p.add('--bptt_steps', type=int, default=0,
                    help='Total length of training trajectories. Must be 0 or a multiple of trajlen')

    p.add('--resample', type=bool, default=False,
                help='add resampling. only works with brains with sequential construction')
    p.add('--alpha', type=float, default=1.0,
                    help='Ratio for soft-resampling. 1.0 default meaning proper resampling')
    p.add('--transition_std', nargs='*', default=["0", "0"],
                help='Standard deviations for motion update. [translate, rotate1(deg), rotate2(deg, optional)]')
    p.add('--init_particles_std', nargs='*', default=["7.07", "6.708"],  #these are the sqrt of defaults in gen_data
                help='Standard deviations for generated initial particles. [translate, rotate(deg)]')

    p.add('--num_particles', type=int,
                    help='Number of particles. Same for each sample')
    p.add('--learningrate', type=float,
                    help='Learning rate')
    p.add('--decaystep', type=int,
                    help='Steps in between decays. For patience this is the number of decays alltogether')
    p.add('--decayrate', type=float,
                    help='Rate of learning rate decay')

    p.add('--gpu', type=int, default=0,
                help='Select gpu devices(cuda index from 0). Default -1, lowest utilization. No gpu if <-1')

    p.add('--logpath', type=str, default='',
                    help='Specify path for logs. Make new by default')
    p.add('--seed', type=int, help='fix numpy and tf seed to this. random if unset')
    p.add('--testseed', type=int, help='fix np and tf seed for test, bot not for train')

    params = p.parse_args(args=args)

    if params.seed is not None and params.seed >= 0:
        np.random.seed(params.seed)

    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    return params