from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, tqdm
import tensorflow as tf
import numpy as np

import pfnet
from arguments import parse_args
from preprocess import get_dataflow

try:
    import ipdb as pdb
except Exception:
    import pdb


def run_evaluation(params):

    # overwrite for evaluation
    params.batchsize = 1
    params.bptt_steps = params.trajlen

    with tf.Graph().as_default():
        if params.seed is not None:
            tf.set_random_seed(params.seed)

        # test data and network
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            test_data, num_test_samples = get_dataflow(params.testfiles, params, is_training=False)
            test_brain = pfnet.PFNet(inputs=test_data[1:], labels=test_data[0], params=params, is_training=False)

        # Add the variable initializer Op.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=3)

        # Create a session for running Ops on the Graph.
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%int(params.gpu)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        # training session
        with tf.Session(config=sess_config) as sess:

            sess.run(init_op)

            # load model from checkpoint file
            if params.load:
                print("Loading model from " + params.load)
                saver.restore(sess, params.load)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            rmse_list = []
            success_list = []

            try:
                for step_i in tqdm.tqdm(range(num_test_samples)):
                    all_distance2, _ = sess.run([test_brain.all_distance2_op, test_brain.update_state_op])

                    # we have squared differences along the trajectory
                    rmse = np.sqrt(np.mean(all_distance2[0]))
                    rmse_list.append(rmse)

                    successful = np.all(np.sqrt(all_distance2[0])[-params.trajlen//4:] < 1.0)
                    success_list.append(successful)

            except KeyboardInterrupt:
                pass

            except tf.errors.OutOfRangeError:
                print("data exhausted")

            finally:
                coord.request_stop()
            coord.join(threads)

            print("RMSE = %fcm" % (np.mean(rmse_list) * 100))
            print("Success rate = %f%%" % (np.mean(np.array(success_list, 'i')) * 100))


if __name__ == '__main__':
    params = parse_args()

    run_evaluation(params)