#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines jobs, intended as units of work.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc
from collections import defaultdict

from ..utilities import nxrange as rge
from ..meta import nxdescriptors as nxd
from ..io import redis as nxr
from . import logger as LOGGER

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class JobTask(nxd.NxAttribute):
    __registry__ = '__tasks__'

    def __init__(self, task_type, input_store=None, output_store=None):
        self.task_type = task_type
        super().__init__(type_=task_type)
        self.input_store = input_store
        self.output_store = output_store

    def __set__(self, job, task):
        try:
            super().__set__(job, task)
        except:
            if job.logger is None:
                raise
            else:
                msg = f"Cannot set task {self.name} for {job}."
                self.logger.exception(msg, exc_info=True)
        task.input_store = self.input_store or job.input_store
        task.output_store = self.output_store or job.output_store


class JobType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        JobTask.collect(cls, namespace)


class Job(metaclass=JobType):
    __etype__ = None
    __input_store__ = None
    __output_store__ = None

    def __init__(self, entity, dt_range=None, input_store=None,
                 output_store=None, stream_mode=False, logger=LOGGER,
                 **params):
        try:
            assert isinstance(entity, self.__etype__)
            assert hasattr(entity, 'id')
        except AssertionError:
            msg = f"Improper entity {entity} supplied to " + \
                  f"{type(self).__name__} job."
            raise TypeError(msg)
        self.entity = entity
        self.stream_mode = stream_mode
        if stream_mode:
            self.dt_range = rge.time_range((None, None))
        else:
            self.dt_range = rge.time_range(dt_range)
        self.input_store = input_store or self.__input_store__
        self.output_store = output_store or self.__output_store__
        self.logger = logger
        self.inputs = defaultdict(dict)
        for name, desc in self.__tasks__.items():
            task = desc.task_type(entity, dt_range, stream_mode=stream_mode,
                                  logger=logger)
            setattr(self, name, task)
            for ti in task.__inputs__.values():
                self.inputs[task.input_store][ti] = None
            for to in task.__outputs__.values():
                self.inputs[task.output_store][to] = None
        self.outputs = dict()

    def retrieve_inputs(self):
        for store, descriptors in self.inputs.items():
            if store is None:
                continue
            elif isinstance(store, nxr.RedisStore):
                pipe = nxr.NxPipe(store)
            else:
                pipe = None
            for d in list(descriptors):
                rval = d.fetch(self.entity, nxpipe=pipe,
                               store=store, dt_range=self.dt_range)
                if rval is not None:
                    descriptors[d] = rval
            if pipe is not None:
                for d, r in zip(list(descriptors), pipe.execute()):
                    descriptors[d] = r

    @classmethod
    def setup_streaming(cls, entity, input_store, output_store, logger=LOGGER):
        for name, desc in cls.__tasks__.items():
            in_store = desc.input_store or input_store
            out_store = desc.output_store or output_store
            desc.task_type.setup_streaming(entity, in_store, out_store,
                                           logger=LOGGER)

    def __call__(self):
        """Run interface."""
        self.retrieve_inputs()
        pipes = {store: nxr.NxPipe(store) for store in self.inputs
                 if isinstance(store, nxr.RedisStore)}
        for name in self.__tasks__:
            task = getattr(self, name)
            in_store = task.input_store
            out_store = task.output_store
            for k, v in task.__inputs__.items():
                setattr(task, k, self.inputs[in_store][v])
            for k, v in task.__outputs__.items():
                setattr(task, k, self.inputs[out_store][v])
            self.outputs[name] = task(nxpipes=pipes)
        for p in pipes.values():
            p.execute()
