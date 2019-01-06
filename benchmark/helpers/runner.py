# -*- coding: utf-8 -*-
"""
| **@created on:** 1/5/19,
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
from itertools import count
import time
from tqdm import tqdm
from benchmark.utils import *
from statistics import mean
from benchmark.report_manager import REPORT
from functools import partial, wraps
from tensorflow.python.client import timeline


def runner_monitor(f, always: bool = False):
    """
    | **@author:** Prathyush SP
    |
    :param f: Function
    :return: Function Return Parameter
    """

    if f is None:
        return partial(runner_monitor, always=always)

    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            print('Starting Runner . . .')
            REPORT.runner_start_memory = REPORT.process.memory_info().rss // 10 ** 6
            REPORT.runner_start_time = time.time()
            r = f(*args, **kwargs)
            REPORT.runner_end_time = time.time()
            REPORT.runner_end_memory = REPORT.process.memory_info().rss // 10 ** 6
            print('Runner Completed Successfully!')
            return r
        except Exception as e:
            print('Runner Failed!')
            raise e

    return wrapped


@runner_monitor
def runner(session_config, dataset_iterator, epoch, fetch_ops):
    _counter = count()

    # Session Creation
    REPORT.session_creation_start_time = time.time()
    sess = tf.Session(config=session_config)
    REPORT.session_creation_end_time = time.time()

    # Session Initialization
    REPORT.session_init_start_time = time.time()
    sess.run(tf.group([tf.global_variables_initializer(), tf.local_variables_initializer(),
                       tf.tables_initializer()]))
    REPORT.session_init_end_time = time.time()

    # Dataset Iterator Initialization
    REPORT.dataset_iterator_init_start_time = time.time()
    sess.run(dataset_iterator.initializer)
    REPORT.dataset_iterator_init_end_time = time.time()
    for i in tqdm(range(epoch)):
        try:
            while True:
                _start = time.time()
                sess.run(fetch_ops)
                REPORT.step_times.append(time.time() - _start)
                print(next(_counter))
        except tf.errors.OutOfRangeError:
            print(f'Epoch {i} completed . . .')
            sess.run(dataset_iterator.initializer)


@runner_monitor
def prop_based_runner(session_config, dataset_iterator, epoch, fetch_ops):
    _counter = count()
    # Session Creation
    REPORT.session_creation_start_time = time.time()
    sess = tf.Session(config=session_config)
    REPORT.session_creation_end_time = time.time()

    # Session Initialization
    REPORT.session_init_start_time = time.time()
    sess.run(tf.group([tf.global_variables_initializer(), tf.local_variables_initializer(),
                       tf.tables_initializer()]))
    REPORT.session_init_end_time = time.time()

    # Dataset Iterator Initialization
    REPORT.dataset_iterator_init_start_time = time.time()
    sess.run(dataset_iterator.initializer)
    REPORT.dataset_iterator_init_end_time = time.time()

    compute_op, apply_op, placeholder_gradients, loss = fetch_ops
    for i in tqdm(range(epoch * 2)):
        try:
            while True:
                _start = time.time()
                REPORT.forward_pass_times.append(timeit_fn(sess.run, fetches=fetch_ops[-1])[-1])
                gradients, _t = timeit_fn(sess.run, fetches=compute_op)
                REPORT.compute_gradient_times.append(_t)
                feed_dict = {placeholder_gradients[e][0]: gradients[e][1] for e, _ in enumerate(compute_op)}
                REPORT.apply_gradients_times.append(timeit_fn(sess.run, fetches=apply_op, feed_dict=feed_dict)[-1])
                REPORT.step_times.append(time.time() - _start)
                print(next(_counter))
        except tf.errors.OutOfRangeError:
            print(f'Epoch {i} completed . . .')
            sess.run(dataset_iterator.initializer)


@runner_monitor
def profiled_runner(session_config, dataset_iterator, epoch, fetch_ops):
    _counter = count()
    _counter = count()
    # Session Creation
    REPORT.session_creation_start_time = time.time()
    sess = tf.Session(config=session_config)
    REPORT.session_creation_end_time = time.time()

    # Session Initialization
    REPORT.session_init_start_time = time.time()
    sess.run(tf.group([tf.global_variables_initializer(), tf.local_variables_initializer(),
                       tf.tables_initializer()]))
    REPORT.session_init_end_time = time.time()

    # Dataset Iterator Initialization
    REPORT.dataset_iterator_init_start_time = time.time()
    sess.run(dataset_iterator.initializer)
    REPORT.dataset_iterator_init_end_time = time.time()

    opts = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
    opts['account_type_regexes'] = ['.*']

    sess.run([tf.global_variables_initializer(), dataset_iterator.initializer])
    profiler = tf.contrib.tfprof.model_analyzer.Profiler(sess.graph)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    for i in tqdm(range(epoch)):
        try:
            while True:
                _start = time.time()
                sess.run(fetch_ops, options=options, run_metadata=run_metadata)
                stp = next(_counter)
                print(stp)
                profiler.add_step(stp, run_metadata)
                with open('{}/timeline_{}_{}.json'.format(REPORT.config['save_path'] + '/timeline/', i, stp), 'w') as f:
                    f.write(timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format())
                REPORT.step_times.append(time.time() - _start)
        except tf.errors.OutOfRangeError:
            print(f'Epoch {i} completed . . .')
            sess.run(dataset_iterator.initializer)
    with open('{}/complete_profile.pctx'.format(REPORT.config['save_path']), 'wb') as f:
        f.write(profiler.serialize_to_string())
