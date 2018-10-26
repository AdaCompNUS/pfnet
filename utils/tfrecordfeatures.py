import tensorflow as tf


def tf_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def tf_bytelist_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def tf_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))