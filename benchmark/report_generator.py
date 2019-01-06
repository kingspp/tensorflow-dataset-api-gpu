# -*- coding: utf-8 -*-
"""
| **@created on:** 1/6/19,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** --
|
"""
import os
import sys
import json
from benchmark.utils import humanize_time_delta
from statistics import mean

try:
    config = json.load(open(sys.argv[1]))
    config['save_path'] += '/' + sys.argv[1].split('/')[-1].split('.')[0] + '/'
    normal_run_report = json.load(open(config['save_path'] + 'normal_run_report.json'))
    profiled_run_report = json.load(open(config['save_path'] + 'profiled_run_report.json'))
    separate_run_report = json.load(open(config['save_path'] + 'separate_run_report.json'))
    normal_profile = json.load(open(config['save_path'] + 'normal_run.json'))
    profiled_profile = json.load(open(config['save_path'] + 'profiled_run.json'))
    separate_profile = json.load(open(config['save_path'] + 'separate_run.json'))
except Exception:
    print('Provide config file path and runner type')
    exit(1)


class ConsolidatedReport(object):
    def __init__(self, config, normal_run, profiled_run, separate_run, normal_profile, profiled_profile,
                 separate_profile):
        """
        Primary View
        """
        self.input_data_size = config['input_size']
        self.epoch = config['epoch']
        self.steps_per_epoch = config['steps']
        self.batch_size = config['batch_size']
        self.prefetch = config['prefetch']
        self.prefetch_to_device = config['prefetch_to_device']
        self.model_size_analytical = normal_run['model_analytical_memory']
        self.elapsed_time = humanize_time_delta(normal_profile['total_elapsed_time (secs)'])
        self.max_gpu_memory_consumption = normal_profile['device_statistics']['gpu_monitor'][
            'max_memory_usage'] if 'gpu_monitor' in normal_profile['device_statistics'] else None
        self.max_gpu_memory_utilization = normal_profile['device_statistics']['gpu_monitor'][
            'max_gpu_usage'] if 'gpu_monitor' in normal_profile['device_statistics'] else None
        self.max_cpu_utilization = normal_profile['device_statistics']['cpu_monitor']['max_cpu_usage (%)']
        self.max_memory_utilization = normal_profile['device_statistics']['memory_monitor']['max_memory_usage (MB)']
        self.activation_parameters = normal_run['activation_parameters']
        self.variable_parameters = normal_run['variable_parameters']
        self.generator_function_calls = normal_run['generator_function_calls']

        """
        Device Statistics
        """
        self.total_available_memory = normal_profile['device_statistics']['memory_monitor']['total_memory (GB)']
        self.cpu_cores = normal_profile['device_statistics']['cpu_monitor']['cpu_cores']
        self.cpu_logic_threads = normal_profile['device_statistics']['cpu_monitor']['cpu_threads']
        self.gpu_stats = None
        self.max_gpu_memory_consumption = normal_profile['device_statistics']['gpu_monitor'][
            'max_memory_usage'] if 'gpu_monitor' in normal_profile['device_statistics'] else None
        self.max_gpu_memory_utilization = normal_profile['device_statistics']['gpu_monitor'][
            'max_gpu_usage'] if 'gpu_monitor' in normal_profile['device_statistics'] else None
        self.max_cpu_utilization = normal_profile['device_statistics']['cpu_monitor']['max_cpu_usage (%)']
        self.max_memory_utilization = normal_profile['device_statistics']['memory_monitor']['max_memory_usage (MB)']

        """
        Time Statistics        
        """
        self.elapsed_time = humanize_time_delta(normal_profile['total_elapsed_time (secs)'])
        self.model_creation_time = humanize_time_delta(normal_run['model_creation_time'])
        self.runner_elapsed_time = humanize_time_delta(normal_run['runner_elapsed_time'])
        self.session_creation_elapsed_time = humanize_time_delta(normal_run['session_creation_elapsed_time'])
        self.session_init_elapsed_time = humanize_time_delta(normal_run['session_init_elapsed_time'])
        self.dataset_iterator_init_elapsed_time = humanize_time_delta(normal_run['dataset_iterator_init_elapsed_time'])
        self.average_step_time_normal_run = humanize_time_delta(normal_run['average_step_time'])
        self.average_step_time_profiled_run = humanize_time_delta(profiled_run['average_step_time'])
        self.average_step_time_separate_run = humanize_time_delta(separate_run['average_step_time'])
        self.average_forward_pass_time = humanize_time_delta(separate_run['average_forward_pass_time'])
        self.average_backward_pass_time = humanize_time_delta(separate_run['average_backward_pass_time'])
        self.average_compute_gradient_time = humanize_time_delta(separate_run['average_compute_gradient_time'])
        self.average_apply_gradients_time = humanize_time_delta(separate_run['average_apply_gradients_time'])
        self.average_iterator_get_next_time = None

        """
        Memory Statistics
        """
        self.model_size_analytical = normal_run['model_analytical_memory']
        self.model_memory_consumption = normal_run['model_creation_memory']
        self.runner_memory_consumption = normal_run['runner_memory_consumption']

    def fetch_details_from_timelines(self, timeline_dir='/private/tmp/time_dirs/sample_config/timeline/'):
        _iterator_get_next_times = []
        for root, dirs, files in os.walk(timeline_dir):
            for f in files:
                d = json.load(open(timeline_dir + f))
                for its in d['traceEvents']:
                    if its['name'] == 'IteratorGetNext' and its['cat'] == 'Op':
                        _iterator_get_next_times.append(its['dur'])
        self.average_iterator_get_next_time = mean(_iterator_get_next_times)

    # Device
    # to
    # Host(Mem
    # Copy)
    # Host
    # to
    # Device(Mem
    # Copy)
