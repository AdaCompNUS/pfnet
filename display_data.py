from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import numpy as np

# Fix Python 2.x.
try: input = raw_input
except NameError: pass

from utils.tfrecordfeatures import *
from preprocess import decode_image, raw_images_to_array

try:
    import ipdb as pdb
except Exception:
    import pdb


def display_data(file):
    gen = tf.python_io.tf_record_iterator(file)
    for data_i, string_record in enumerate(gen):
        result = tf.train.Example.FromString(string_record)
        features = result.features.feature

        # maps are np.uint8 arrays. each has a different size.

        # wall map: 0 for free space, 255 for walls
        map_wall = decode_image(features['map_wall'].bytes_list.value[0])

        # door map: 0 for free space, 255 for doors
        map_door = decode_image(features['map_door'].bytes_list.value[0])

        # roomtype map: binary encoding of 8 possible room categories
        # one state may belong to multiple room categories
        map_roomtype = decode_image(features['map_roomtype'].bytes_list.value[0])

        # roomid map: pixels correspond to unique room ids.
        # for overlapping rooms the higher ids overwrite lower ids
        map_roomid = decode_image(features['map_roomid'].bytes_list.value[0])

        # true states
        # (x, y, theta). x,y: pixel coordinates; theta: radians
        # coordinates index the map as a numpy array: map[x, y]
        true_states = features['states'].bytes_list.value[0]
        true_states = np.frombuffer(true_states, np.float32).reshape((-1, 3))

        # odometry
        # each entry is true_states[i+1]-true_states[i].
        # last row is always [0,0,0]
        odometry = features['odometry'].bytes_list.value[0]
        odometry = np.frombuffer(odometry, np.float32).reshape((-1, 3))

        # observations are enceded as a list of png images
        rgb = raw_images_to_array(list(features['rgb'].bytes_list.value))
        depth = raw_images_to_array(list(features['depth'].bytes_list.value))

        print ("True states (first three)")
        print (true_states[:3])

        print ("Odometry (first three)")
        print (odometry[:3])

        print("Plot map and first observation")

        # note: when printed as an image, map should be transposed
        plt.figure()
        plt.imshow(map_wall.transpose())

        plt.figure()
        plt.imshow(rgb[0])

        plt.show()

        if input("proceed?") != 'y':
            break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage: display_data.py xxx.tfrecords")
        exit()

    display_data(sys.argv[1])