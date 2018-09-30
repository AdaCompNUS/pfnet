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


def validation(sess, brain, num_samples, params):

    fix_seed = (params.testseed is not None and params.testseed >= 0)
    if fix_seed:
        np_random_state = np.random.get_state()
        np.random.seed(params.testseed)
        tf.set_random_seed(params.testseed)

    saved_state = brain.save_state(sess)

    total_loss = 0.0
    try:
        for eval_i in tqdm.tqdm(range(num_samples), desc="Validation"):
            loss, _ = sess.run([brain.valid_loss_op, brain.update_state_op])
            total_loss += loss

        print ("Validation loss = %f"%(total_loss/num_samples))

    except tf.errors.OutOfRangeError:
        print ("No more samples for evaluation. This should not happen")
        raise

    brain.load_state(sess, saved_state)

    # restore seed
    if fix_seed:
        np.random.set_state(np_random_state)
        tf.set_random_seed(np.random.randint(999999))  # cannot save tf seed, so generate random one from numpy

    return total_loss


def run_training(params):
    with tf.Graph().as_default():
        if params.seed is not None:
            tf.set_random_seed(params.seed)

        # training data and network
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_data, num_train_samples = get_dataflow(params.trainfiles, params, is_training=True)
            train_brain = pfnet.PFNet(inputs=train_data[1:], labels=train_data[0], params=params, is_training=True)

        # test data and network
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
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

            try:
                decay_step = 0
                for epoch_i in range(params.epochs):
                    epoch_loss = 0.0
                    periodic_loss = 0.0

                    # #TODO remove
                    # loss, eval_loss, particle_states, particle_weights, true_states, init_particles = sess.run([train_brain.train_loss_op, train_brain.eval_loss_op, train_brain.outputs[0], train_brain.outputs[1], train_data[0], train_data[2]])
                    # print (loss)
                    # particle_mean = np.mean(particle_states, axis=1)
                    # print (particle_mean[0, :, :2])
                    # print (true_states[0, :, :2])
                    # print (init_particles[0])
                    # pdb.set_trace()

                    # validation(sess, test_brain, num_samples=num_test_samples, params=params)

                    for step_i in tqdm.tqdm(range(num_train_samples), leave=False):
                        _, loss, _ = sess.run([train_brain.train_op, train_brain.train_loss_op,
                                               train_brain.update_state_op])
                        periodic_loss += loss
                        epoch_loss += loss

                        # print accumulated loss after every few steps
                        if step_i > 0 and (step_i % 500) == 0:
                            print("Training loss = %f" % (periodic_loss / 500.0))
                            periodic_loss = 0.0

                    print("Epoch training loss = %f" % (epoch_loss / num_train_samples))

                    # save model, validate and decrease learning rate after each epoch
                    saver.save(sess, os.path.join(params.logpath, 'model.chk'), global_step=epoch_i)

                    validation(sess, test_brain, num_samples=num_test_samples, params=params)

                    #  decay learning rate
                    if epoch_i > 0 and (epoch_i % params.decaystep == 0):
                        decay_step += 1
                        current_global_step = sess.run(tf.assign(train_brain.global_step_op, decay_step))
                        current_learning_rate = sess.run(train_brain.learning_rate_op)
                        print("Decreased learning rate to %f." % (current_learning_rate))

            except KeyboardInterrupt:
                pass

            except tf.errors.OutOfRangeError:
                print("data exhausted")

            finally:
                saver.save(sess, os.path.join(params.logpath, 'final.chk'))  # dont pass global step
                coord.request_stop()

            coord.join(threads)

        print ("Saved to %s"%(params.logpath))



if __name__ == '__main__':
    params = parse_args()

    run_training(params)