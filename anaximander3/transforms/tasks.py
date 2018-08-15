#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines tasks, which specify io as well as transforms.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc

from ..utilities import xprops, nxrange as rge
from ..data import datalogs as dtl
from ..io import redis
from . import logger as LOGGER
from .exceptions import InputError

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class Task(abc.ABC):
    __etype__ = None
    __inputs__ = {}
    __output__ = None

    def __init__(self, entity, dt_range=None, input_store=None,
                 output_store=None, logger=LOGGER, **inputs):
        try:
            assert isinstance(entity, self.__etype__)
            assert hasattr(entity, 'id')
        except AssertionError:
            msg = f"Improper entity {entity} supplied to " + \
                  f"{type(self).__name__} task."
            raise InputError(msg)
        self.entity = entity
        self.dt_range = rge.time_range(dt_range)
        self.input_store = input_store
        self.output_store = output_store
        self.logger = logger
        for k, v in inputs.items():
            try:
                title = self.__inputs__[k]
            except KeyError:
                msg = f"Unknown input {k}."
                raise InputError(msg)
            try:
                assert isinstance(v.schema, type(title.schema))
            except AssertionError:
                msg = f"Invalid input {v}, must be compatible with {title}."
                raise InputError(msg)
        self._inputs = inputs

    @xprops.cachedproperty
    def inputs(self):
        return {}

    def __getattr__(self, attr):
        try:
            return self._inputs[attr]
        except KeyError:
            raise AttributeError

    @xprops.cachedproperty
    def output(self):
        return None

    @classmethod
    def query_input(cls, input_store, title, entity, dt_range):
        if input_store is None:
            msg = "An input store must be specified to retrieve task inputs"
            raise TypeError(msg)
        elif isinstance(input_store,
                        (redis.RedisProcessStore,
                         redis.RedisApplicationStore)):
            tract = input_store[title]
            return tract.sequence(entity.id)
        else:
            tract = input_store[title]
            query = tract.query(id=entity.id, datetime=dt_range)
            return query.data()

    def retrieve_inputs(self):
        for name, title in self.__inputs__.items():
            if name in self._inputs:
                continue
            self._inputs[name] = self.query_input(self.input_store,
                                                  title,
                                                  self.entity,
                                                  self.dt_range)

    @abc.abstractmethod
    def __task__(self):
        """Type-specific task method."""
        return None

    def __call__(self, plot=False):
        """Run interface."""
        self.retrieve_inputs()
        try:
            output = self.__task__()
            assert isinstance(output, dtl.DataSequence)
            assert isinstance(output.schema, type(self.__output__.schema))
        except:
            if self.logger is None:
                raise
            msg = f"{type(self).__name__} task error on {self.inputs}."
            self.logger.exception(msg, exc_info=True)
            return None
        self._output = output
        if plot:
            self.plot()
        return output

    def plot(self, **kwargs):
        if self.output is None:
            self()
        self.__plot__(**kwargs)

    def __plot__(self, **kwargs):
        """Optional operator plot."""
        pass
