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
from tensorflow.contrib import rnn
import typing
import time
from benchmark.report_manager import REPORT
from functools import wraps, partial, reduce
import operator
from benchmark.utils import *


def model_mapper(model: str, *args, **kwargs):
    mmaper = {
        'mnist_ffn': mnist_model_ffn,
        'mnist_cnn': mnist_model_cnn,
        'mnist_rnn': mnist_model_birnn,
    }
    return mmaper[model](*args, **kwargs)


def optimizer_mapper(optimizer: str):
    optmapper = {
        'sgd': tf.train.GradientDescentOptimizer,
        'adam': tf.train.AdamOptimizer
    }
    return optmapper[optimizer]


def update_activation_parameters(tensor):
    REPORT.activation_parameters += reduce(operator.mul, [REPORT.config['batch_size'], *tensor.shape.as_list()[1:]])
    return tensor


def model_monitor(f, always: bool = False):
    """
    | **@author:** Prathyush SP
    |
    | Key Exception Decorator.
    :param f: Function
    :return: Function Return Parameter
    """

    if f is None:
        return partial(model_monitor, always=always)

    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            print('Creating Model . . .')
            tf.Graph().__init__()
            REPORT.model_start_memory = REPORT.process.memory_info().rss // 10 ** 6
            REPORT.model_start_time = time.time()
            r = f(*args, **kwargs)
            REPORT.model_end_time = time.time()
            REPORT.model_end_memory = REPORT.process.memory_info().rss // 10 ** 6
            REPORT.variable_parameters = sum([reduce(operator.mul, t.get_shape().as_list() if len(t.get_shape().as_list())>1 else [0]) for t in
                                              tf.get_default_graph().get_collection(
                                                  tf.GraphKeys.GLOBAL_VARIABLES)])
            REPORT.model_analytical_memory += get_param_size(REPORT.activation_parameters) * 2
            REPORT.model_analytical_memory += get_param_size(REPORT.variable_parameters) * 3
            print('Model Created Successfully!')
            return r
        except Exception as e:
            print('Model Created Failed!')
            raise e

    return wrapped


def create_fetch_ops(f_type, optimizer: tf.train.Optimizer, loss):
    if f_type == 'separate':
        compute_op = optimizer.compute_gradients(loss=loss)
        placeholder_gradients = [(tf.placeholder(dtype=tf.float32, shape=grad.shape), var) for grad, var in compute_op]
        apply_op = optimizer.apply_gradients(grads_and_vars=placeholder_gradients)
        return compute_op, apply_op, placeholder_gradients, loss
    else:
        return optimizer.minimize(loss), loss

@model_monitor
def mnist_model_birnn(features, target, optimizer: str, num_hidden: int = 128, num_classes=10,
                      f_type=''):
    features = update_activation_parameters(features)
    target = update_activation_parameters(target)
    weights = {
        'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    x = [update_activation_parameters(i) for i in tf.unstack(features, features.get_shape()[1], 1)]
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                     dtype=tf.float32)
        outputs = [update_activation_parameters(i) for i in outputs]
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                               dtype=tf.float32)
        outputs[-1] = update_activation_parameters(outputs[-1])
    logits = update_activation_parameters(tf.matmul(outputs[-1], weights['out']) + biases['out'])
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=target))
    optimizer = optimizer_mapper(optimizer)(learning_rate=0.01)
    return create_fetch_ops(optimizer=optimizer, loss=loss_op, f_type=f_type)

@model_monitor
def mnist_model_cnn(features, target, optimizer: str, f_type=''):
    x = tf.layers.conv2d(inputs=features, filters=30, kernel_size=(5, 5), activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2))
    x = tf.layers.conv2d(inputs=x, filters=15, kernel_size=(3, 3), activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(inputs=x, pool_size=(2, 2), strides=(2, 2))
    x = tf.layers.dropout(inputs=x, rate=0.2)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=50, activation=tf.nn.relu)
    x = tf.layers.dense(inputs=x, units=10, activation=tf.nn.softmax)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=target))
    optimizer = optimizer_mapper(optimizer)(learning_rate=0.01)
    return create_fetch_ops(optimizer=optimizer, loss=loss_op, f_type=f_type)


@model_monitor
def mnist_model_ffn(features, target, optimizer: str, f_type=''):
    features = update_activation_parameters(features)
    target = update_activation_parameters(target)
    x = update_activation_parameters(tf.layers.dense(inputs=features, units=500, activation=tf.nn.relu))
    x = update_activation_parameters(tf.layers.dense(inputs=x, units=100, activation=tf.nn.relu))
    x = update_activation_parameters(tf.layers.dense(inputs=x, units=10))
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=x, labels=target))
    optimizer = optimizer_mapper(optimizer)(learning_rate=0.01)
    return create_fetch_ops(optimizer=optimizer, loss=loss_op, f_type=f_type)
