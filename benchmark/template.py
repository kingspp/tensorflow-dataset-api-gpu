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

from pmark import pmonitor
from pmark.writers import JSONWriter
from pmark.monitors import CPUMonitor, GPUMonitor, MemoryMonitor
from benchmark.helpers.model import *
from benchmark.helpers.data import *
from benchmark.helpers.runner import *
from benchmark.report_manager import REPORT
import json
import sys
import os

try:
    config = json.load(open(sys.argv[1]))
    config['name'] = sys.argv[1].split('/')[-1].split('.')[0]
    config['save_path'] += '/' + config['name'] + '/'
    os.system('mkdir -p {}'.format(config['save_path']))
    os.system('mkdir -p {}'.format(config['save_path'] + '/timeline/'))
    runner_type = sys.argv[2]
except Exception:
    print('Provide config file path and runner type')
    exit(1)


@pmonitor(f=None, monitors=[CPUMonitor, GPUMonitor, MemoryMonitor],
          writers=[JSONWriter(save_path=config['save_path'], file_name='normal_run.json')])
def normal_run(config):
    REPORT.config = config
    iterator, features, target = dataset_generator_inmem(batch_size=config['batch_size'], steps=config['steps'],
                                                         prefetch=config['prefetch'],
                                                         prefetch_to_device=config['prefetch_to_device'],
                                                         features_shape=config['input_dimensions']['features'],
                                                         label_shape=config['input_dimensions']['labels'])
    fetch_ops = model_mapper(model=config['model'], features=features, optimizer=config['optimizer'], target=target)

    session_config = tf.ConfigProto(log_device_placement=config['session_config']['log_device_placement'],
                                    allow_soft_placement=config['session_config']['allow_soft_placement'])
    session_config.gpu_options.allow_growth = config['session_config']['allow_growth']
    runner(session_config=session_config, fetch_ops=fetch_ops, epoch=config['epoch'],
           dataset_iterator=iterator)
    REPORT.complete_aggregates()
    json.dump(REPORT.__dict__, open(config['save_path'] + '/normal_run_report.json', 'w'), indent=2)


@pmonitor(f=None, monitors=[CPUMonitor, GPUMonitor, MemoryMonitor],
          writers=[JSONWriter(save_path=config['save_path'], file_name='profiled_run.json')])
def profiled_run(config):
    REPORT.config = config
    iterator, features, target = dataset_generator_inmem(batch_size=config['batch_size'], steps=config['steps'],
                                                         prefetch=config['prefetch'],
                                                         prefetch_to_device=config['prefetch_to_device'],
                                                         features_shape=config['input_dimensions']['features'],
                                                         label_shape=config['input_dimensions']['labels'])
    fetch_ops = model_mapper(model=config['model'], features=features, optimizer=config['optimizer'], target=target)

    session_config = tf.ConfigProto(log_device_placement=config['session_config']['log_device_placement'],
                                    allow_soft_placement=config['session_config']['allow_soft_placement'])
    session_config.gpu_options.allow_growth = config['session_config']['allow_growth']
    profiled_runner(session_config=session_config, fetch_ops=fetch_ops, epoch=config['epoch'],
                    dataset_iterator=iterator)
    REPORT.complete_aggregates()
    json.dump(REPORT.__dict__, open(config['save_path'] + '/profiled_run_report.json', 'w'), indent=2)


@pmonitor(f=None, monitors=[CPUMonitor, GPUMonitor, MemoryMonitor],
          writers=[JSONWriter(save_path=config['save_path'], file_name='separate_run.json')])
def separate_run(config):
    REPORT.config = config
    iterator, features, target = dataset_generator_inmem(batch_size=config['batch_size'], steps=config['steps'],
                                                         prefetch=config['prefetch'],
                                                         prefetch_to_device=config['prefetch_to_device'],
                                                         features_shape=config['input_dimensions']['features'],
                                                         label_shape=config['input_dimensions']['labels'])
    fetch_ops = model_mapper(model=config['model'], features=features, target=target, optimizer=config['optimizer'],
                             f_type='separate')
    session_config = tf.ConfigProto(log_device_placement=config['session_config']['log_device_placement'],
                                    allow_soft_placement=config['session_config']['allow_soft_placement'])
    session_config.gpu_options.allow_growth = config['session_config']['allow_growth']
    prop_based_runner(session_config=session_config, fetch_ops=fetch_ops, epoch=config['epoch'],
                      dataset_iterator=iterator)
    REPORT.complete_aggregates()
    json.dump(REPORT.__dict__, open(config['save_path'] + '/separate_run_report.json', 'w'), indent=2)


if runner_type == 'normal':
    normal_run(config=config)
elif runner_type == 'profiled':
    profiled_run(config=config)
elif runner_type == 'separate':
    separate_run(config=config)
else:
    print('Unknown run configuration. Supported are `normal`, `profiled`, `separate`')

    #
    # sample_config = {
    #     'name': 'run_1',
    #     'save_path': '/tmp/timedrs/',
    #     'epoch': 10,
    #     'batch_size': 32,
    #     'steps': 10,
    #     'optimizer': 'sgd',
    #     'prefetch': 10,
    #     'prefetch_to_device': None,
    #     'model': 'mnist_ffn',
    #     'runner': 'prof_based',
    #     'input_dimensions': {
    #         'features': [784],
    #         'labels': [10]
    #     },
    #     'session_config': {
    #         "allow_growth": True,
    #         "log_device_placement": False,
    #         "allow_soft_placement": True,
    #     }
    #
    # }

    # normal_run(config=sample_config)
    # main_profile_runner(config=v)
    # main_prop_based_runner(config=v)
