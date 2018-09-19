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
import time

from ..utilities import nxrange as rge, functions as fun
from ..meta import nxdescriptors as nxd
from ..io import redis as nxr
from . import logger as LOGGER
from .exceptions import OperationalError

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
            msg = f"Cannot set task {self.name} for {job}."
            self.logger.exception(msg)
            raise OperationalError

    def __call__(self, job):
        task = self.task_type(job.entity, job.dt_range,
                              stream_mode=job.stream_mode,
                              logger=job.logger,
                              input_store=self.input_store,
                              output_store=self.output_store)
        setattr(job, self.name, task)
        return task


class JobType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        JobTask.collect(cls, namespace)


class Job(metaclass=JobType):
    __etype__ = None
    __input_store__ = None
    __output_store__ = None

    def __init__(self, entity, dt_range=None, stream_mode=False, logger=LOGGER,
                 **params):
        self.entity = entity
        self.stream_mode = stream_mode
        if stream_mode:
            self.dt_range = rge.time_range((None, None))
        else:
            self.dt_range = rge.time_range(dt_range)
        self.logger = logger
        try:
            assert isinstance(entity, self.__etype__)
            assert hasattr(entity, 'store_id')
        except AssertionError:
            msg = f"Improper entity {entity} supplied to " + \
                  f"{type(self).__name__}."
            self.logger.exception(msg)
            raise OperationalError
        self.inputs = defaultdict(dict)
        for name, desc in self.__tasks__.items():
            task = desc(self)
            for ti in task.__inputs__.values():
                self.inputs[task.input_store][ti] = None
            for to in task.__outputs__.values():
                self.inputs[task.output_store][to] = None
        self.outputs = dict()

    def retrieve_inputs(self):
        fetch_count = 0
        for store, descriptors in self.inputs.items():
            if store is None:
                continue
            elif isinstance(store, nxr.RedisStore):
                pipe = nxr.NxPipe(store)
            else:
                pipe = None
            for dsc in list(descriptors):
                rval = dsc.fetch(self.entity, self.dt_range,
                                 store=store, nxpipe=pipe)
                fetch_count += 1
                if rval is not None:
                    descriptors[dsc] = rval
            if pipe is not None:
                for dsc, res in zip(list(descriptors), pipe.execute()):
                    descriptors[dsc] = res
        return fetch_count

    @classmethod
    def setup_streaming(cls, entity, input_store, output_store, logger=LOGGER):
        for name, desc in cls.__tasks__.items():
            in_store = desc.input_store or input_store
            out_store = desc.output_store or output_store
            desc.task_type.setup_streaming(entity, in_store, out_store,
                                           logger=LOGGER)

    def __call__(self):
        """Run interface."""
        t0 = time.time()
        fetch_count = self.retrieve_inputs()
        t1 = time.time()
        run_1 = round(1e3 * (t1 - t0))
        msg = f"{self} retrieved {fetch_count} data inputs " + \
              f"in {run_1:,} milliseconds."
        self.logger.debug(msg)
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
            outputs = task(nxpipes=pipes)
            self.outputs[name] = outputs
            for desc, out in zip(task.__outputs__.values(), outputs):
                self.inputs[out_store][desc] = out
        t2 = time.time()
        run_2 = round(1e3 * (t2 - t1))
        msg = f"{self} ran task opertions in {run_2:,} milliseconds."
        self.logger.debug(msg)
        for p in pipes.values():
            p.execute()
        t3 = time.time()
        run_3 = round(1e3 * (t3 - t2))
        msg = f"{self} uploaded results in {run_3:,} milliseconds."
        self.logger.debug(msg)
        total_run = round(1e3 * (t3 - t0))
        msg = f"{self} completed in {total_run:,} milliseconds."
        self.logger.info(msg)

    def __repr__(self):
        return fun.iformat('entity')(self)
