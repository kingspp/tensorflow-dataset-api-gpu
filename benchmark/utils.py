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
import time


def timeit_fn(fn, **kwargs):
    start = time.time()
    fn_res = fn(**kwargs)
    return fn_res, (time.time() - start)


def humanize_time_delta(td_object):
    seconds = td_object
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('minute', 60),
        ('second', 1),
        ('milli_second', 1 / 10 ** 3),
        ('micro_second', 1 / 10 ** 6),
        ('nano_second', 1 / 10 ** 9),
    ]

    strings = []
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = 's' if period_value > 1 else ''
            strings.append("%s %s%s" % (period_value, period_name, has_s))
    return ", ".join(strings)


get_param_size = lambda params: params * 4 // (1024 ** 2)
