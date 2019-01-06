# -*- coding: utf-8 -*-
"""
| **@created on:** 12/28/18,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** --
|
"""

import tensorflow as tf
import numpy as np
from benchmark.report_manager import REPORT


def dataset_generator_inmem(batch_size, steps, prefetch=None, prefetch_to_device=None, features_shape=[784],
                            label_shape=[10]):
    # from tensorflow.examples.tutorials.mnist import input_data
    # import os, tempfile
    # mnist_save_dir = os.path.join(tempfile.gettempdir(), 'MNIST_data')
    # mnist = input_data.read_data_sets(mnist_save_dir, one_hot=True)
    def _gen():
        REPORT.generator_function_calls += 1
        # for b_data, b_label in zip(mnist.train.images[:batch_size * steps], mnist.train.labels[:batch_size * steps]):
        #     yield b_data, b_label
        for i in range(batch_size * steps):
            yield np.random.random(features_shape), np.random.random(label_shape)

    dataset = tf.data.Dataset.from_generator(generator=_gen, output_types=(tf.float32, tf.float32),
                                             output_shapes=(features_shape, label_shape))
    dataset = dataset.batch(batch_size)

    if prefetch and prefetch_to_device:
        dataset = dataset.apply(
            tf.contrib.data.prefetch_to_device(prefetch_to_device, prefetch))
    elif prefetch:
        dataset = dataset.prefetch(prefetch)
    elif prefetch_to_device:
        dataset = dataset.apply(
            tf.contrib.data.prefetch_to_device(prefetch_to_device, prefetch))

    # Create Dataset Iterator
    iterator = dataset.make_initializable_iterator()

    # Create features and labels
    next_x, next_y = iterator.get_next()
    return iterator, next_x, next_y


def dataset_generator_kafka(prefetch):
    pass


def dataset_tensor_slices():
    pass


def placeholder():
    pass
