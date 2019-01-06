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
__all__ = ['REPORT']

from benchmark.singleton import Singleton
import os
import psutil
import time
from statistics import mean


class _Report(metaclass=Singleton):
    def __init__(self):
        self.name = None
        self.config = None
        self.relative_start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.model_creation_time = None
        self.model_creation_memory = None
        self.model_analytical_memory = 0
        self.activation_parameters = 0
        self.variable_parameters = 0
        self.model_start_time = None
        self.model_end_time = None
        self.model_start_memory = None
        self.model_end_memory = None
        self.model_relative_start_time = None
        self.model_relative_end_time = None
        self.runner_start_time = None
        self.runner_end_time = None
        self.runner_elapsed_time = None
        self.session_creation_start_time = None
        self.session_creation_end_time = None
        self.session_creation_elapsed_time = None
        self.session_init_start_time = None
        self.session_init_end_time = None
        self.session_init_elapsed_time = None
        self.dataset_iterator_init_start_time = None
        self.dataset_iterator_init_end_time = None
        self.dataset_iterator_init_elapsed_time = None
        self.step_times = []
        self.average_step_time = None
        self.forward_pass_times = []
        self.backward_pass_times = []
        self.compute_gradient_times = []
        self.apply_gradients_times = []
        self.average_forward_pass_time = None
        self.average_backward_pass_time = None
        self.average_compute_gradient_time = None
        self.average_apply_gradients_time = None
        self.runner_start_memory = None
        self.runner_end_memory = None
        self.runner_memory_consumption = None
        self.generator_function_calls = 0

    def complete_aggregates(self):
        self.process = None
        self.model_creation_time = self.model_end_time - self.model_start_time
        self.model_creation_memory = self.model_end_memory - self.model_start_memory
        self.model_relative_start_time = self.model_start_time - self.relative_start_time
        self.model_relative_end_time = self.model_relative_start_time + self.model_creation_time
        self.runner_elapsed_time = self.runner_end_time - self.runner_start_time
        self.session_creation_elapsed_time = self.session_creation_end_time - self.session_creation_start_time
        self.session_init_elapsed_time = self.session_init_end_time - self.session_init_start_time
        self.dataset_iterator_init_elapsed_time = self.dataset_iterator_init_end_time - self.dataset_iterator_init_start_time
        self.average_step_time = mean(self.step_times)
        self.runner_memory_consumption = self.runner_end_memory - self.runner_start_memory
        if len(self.forward_pass_times) > 0:
            self.average_forward_pass_time = mean(self.forward_pass_times)

        if len(self.compute_gradient_times) > 0:
            self.average_compute_gradient_time = mean(self.compute_gradient_times)

        if len(self.apply_gradients_times) > 0:
            self.average_apply_gradients_time = mean(self.apply_gradients_times)

        if len(self.forward_pass_times) > 0:
            self.backward_pass_times = [cp + ap for cp, ap in
                                        zip(self.compute_gradient_times, self.apply_gradients_times)]
            self.average_backward_pass_time = mean(self.backward_pass_times)


REPORT = _Report()
