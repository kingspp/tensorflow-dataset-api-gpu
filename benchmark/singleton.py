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

__all__ = ['Singleton']

from abc import ABCMeta


class Singleton(type, metaclass=ABCMeta):
    """
    Singleton Class
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
