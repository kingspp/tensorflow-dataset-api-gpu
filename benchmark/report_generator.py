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

GPU_DEVICES = {
    '0': 'GPU-7da3f67f-ef6f-e43d-2433-ae178b12001d',
    '1': 'GPU-86fde21a-eb59-b7d8-dace-88f81a1cb2ea',
    '0,1': 'GPU-7da3f67f-ef6f-e43d-2433-ae178b12001d',
}


class ConsolidatedReport(object):
    def __init__(self, config, normal_run, profiled_run, separate_run, normal_profile, profiled_profile,
                 separate_profile):
        """
        Primary View
        """
        self.input_data_size = config['input_dimensions']
        self.epoch = config['epoch']
        self.steps_per_epoch = config['steps']
        self.batch_size = config['batch_size']
        self.prefetch = config['prefetch']
        self.prefetch_to_device = config['prefetch_to_device']
        self.model_size_analytical = normal_run['model_analytical_memory']
        self.elapsed_time = humanize_time_delta(normal_profile['total_elapsed_time (secs)'])
        if not config['session_config']['gpu_devices'] == '' and 'gpu_monitor' in normal_profile[
            'device_statistics'] and not 'error' in normal_profile['device_statistics'][
            'gpu_monitor']:
            self.max_gpu_memory_consumption = \
                normal_profile['device_statistics']['gpu_monitor']['gpu_max_memory_usage (in MiB)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_utilization = normal_profile['device_statistics']['gpu_monitor']['gpu_max_utilization (in %)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
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

        self.max_cpu_utilization = normal_profile['device_statistics']['cpu_monitor']['max_cpu_usage (%)']
        self.max_memory_utilization = normal_profile['device_statistics']['memory_monitor']['max_memory_usage (MB)']

        self.max_cpu_utilization_profile = profiled_profile['device_statistics']['cpu_monitor']['max_cpu_usage (%)']
        self.max_memory_utilization_profile = profiled_profile['device_statistics']['memory_monitor'][
            'max_memory_usage (MB)']

        self.max_cpu_utilization_separate = separate_profile['device_statistics']['cpu_monitor']['max_cpu_usage (%)']
        self.max_memory_utilization_separate = separate_profile['device_statistics']['memory_monitor'][
            'max_memory_usage (MB)']

        if not config['session_config']['gpu_devices'] == '' and 'gpu_monitor' in normal_profile[
            'device_statistics'] and not 'error' in normal_profile['device_statistics']['gpu_monitor']:
            self.gpu_stats = None

            # Normal Run
            self.total_gpu_memory = normal_profile['device_statistics']['gpu_monitor']['gpu_total_memory (in MiB)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.gpu_power_limit = normal_profile['device_statistics']['gpu_monitor']['gpu_power_limit (in Watt)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_memory_consumption = \
                normal_profile['device_statistics']['gpu_monitor']['gpu_max_memory_usage (in MiB)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_utilization = normal_profile['device_statistics']['gpu_monitor']['gpu_max_utilization (in %)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.gpu_max_power_drawn = \
                normal_profile['device_statistics']['gpu_monitor']['gpu_max_power_drawn (in Watt)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]

            # Profiled Run
            self.total_gpu_memory_profiled = \
            profiled_profile['device_statistics']['gpu_monitor']['gpu_total_memory (in MiB)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.gpu_power_limit_profiled = \
            profiled_profile['device_statistics']['gpu_monitor']['gpu_power_limit (in Watt)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_memory_consumption_profiled = \
                profiled_profile['device_statistics']['gpu_monitor']['gpu_max_memory_usage (in MiB)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_utilization_profiled = \
            profiled_profile['device_statistics']['gpu_monitor']['gpu_max_utilization (in %)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.gpu_max_power_drawn_profiled = \
                profiled_profile['device_statistics']['gpu_monitor']['gpu_max_power_drawn (in Watt)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]

            # Separate Run
            self.total_gpu_memory_separate = \
            separate_profile['device_statistics']['gpu_monitor']['gpu_total_memory (in MiB)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.gpu_power_limit_separate = \
            separate_profile['device_statistics']['gpu_monitor']['gpu_power_limit (in Watt)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_memory_consumption_separate = \
                separate_profile['device_statistics']['gpu_monitor']['gpu_max_memory_usage (in MiB)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.max_gpu_utilization_separate = \
            separate_profile['device_statistics']['gpu_monitor']['gpu_max_utilization (in %)'][
                GPU_DEVICES[config['session_config']['gpu_devices']]]
            self.gpu_max_power_drawn_separate = \
                separate_profile['device_statistics']['gpu_monitor']['gpu_max_power_drawn (in Watt)'][
                    GPU_DEVICES[config['session_config']['gpu_devices']]]

        """
        Time Statistics        
        """
        # Normal run
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
        self.average_memcpyd2h_time = None
        self.average_memcpyh2d_time = None
        self.total_memcpy_d2h_calls = None
        self.total_memcpy_h2d_calls = None
        self.memcpy_d2h_calls_per_step = None
        self.memcpy_h2d_calls_per_step = None

        # Profiled Run
        self.elapsed_time_profiled = humanize_time_delta(profiled_profile['total_elapsed_time (secs)'])
        self.model_creation_time_profiled = humanize_time_delta(profiled_profile['model_creation_time'])
        self.runner_elapsed_time_profiled = humanize_time_delta(profiled_profile['runner_elapsed_time'])
        self.session_creation_elapsed_time_profiled = humanize_time_delta(
            profiled_profile['session_creation_elapsed_time'])
        self.session_init_elapsed_time_profiled = humanize_time_delta(profiled_profile['session_init_elapsed_time'])
        self.dataset_iterator_init_elapsed_time_profiled = humanize_time_delta(
            profiled_profile['dataset_iterator_init_elapsed_time'])

        # Separate Run
        self.elapsed_time_separate = humanize_time_delta(separate_profile['total_elapsed_time (secs)'])
        self.model_creation_time_separate = humanize_time_delta(separate_profile['model_creation_time'])
        self.runner_elapsed_time_separate = humanize_time_delta(separate_profile['runner_elapsed_time'])
        self.session_creation_elapsed_time_separate = humanize_time_delta(
            separate_profile['session_creation_elapsed_time'])
        self.session_init_elapsed_time_separate = humanize_time_delta(separate_profile['session_init_elapsed_time'])
        self.dataset_iterator_init_elapsed_time_separate = humanize_time_delta(
            separate_profile['dataset_iterator_init_elapsed_time'])

        """
        Memory Statistics
        """
        self.model_size_analytical = normal_run['model_analytical_memory']
        self.model_memory_consumption = normal_run['model_creation_memory']
        self.runner_memory_consumption = normal_run['runner_memory_consumption']

    def fetch_details_from_timelines(self, timeline_dir):
        _iterator_get_next_times, _memcpy_d2h_times, _memcpy_h2d_times = [], [], []
        for root, dirs, files in os.walk(timeline_dir):
            for f in files:
                d = json.load(open(timeline_dir + f))
                for its in d['traceEvents']:
                    if its['name'] == 'IteratorGetNext' and its['cat'] == 'Op':
                        _iterator_get_next_times.append(its['dur'])
                    if its['name'] == 'MEMCPYHtoD' and its['cat'] == 'Op':
                        _memcpy_h2d_times.append(its['dur'])
                    if its['name'] == 'MEMCPYDtoH' and its['cat'] == 'Op':
                        _memcpy_d2h_times.append(its['dur'])

        self.average_iterator_get_next_time = humanize_time_delta(mean(_iterator_get_next_times) / 10 ** 6)
        if not config['session_config']['gpu_devices'] == '' and 'gpu_monitor' in normal_profile[
            'device_statistics'] and not 'error' in normal_profile['device_statistics'][
            'gpu_monitor']:
            self.average_memcpyd2h_time = humanize_time_delta(mean(_memcpy_d2h_times) / 10 ** 6)
            self.average_memcpyh2d_time = humanize_time_delta(mean(_memcpy_h2d_times) / 10 ** 6)
            self.total_memcpy_d2h_calls = len(_memcpy_d2h_times)
            self.total_memcpy_h2d_calls = len(_memcpy_h2d_times)
            self.memcpy_d2h_calls_per_step = len(_memcpy_d2h_times) / (config['epoch'] * config['steps'])
            self.memcpy_h2d_calls_per_step = len(_memcpy_h2d_times) / (config['epoch'] * config['steps'])
        return self


json.dump(ConsolidatedReport(
    config=config,
    normal_run=normal_run_report,
    profiled_run=profiled_run_report,
    separate_run=separate_run_report,
    normal_profile=normal_profile,
    profiled_profile=profiled_profile,
    separate_profile=separate_profile
).fetch_details_from_timelines(timeline_dir=config['save_path'] + '/timeline/').__dict__,
          open(config['save_path'] + '/consolidated_report.json', 'w'), indent=2)
