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
from collections import Sequence

from ..utilities import xprops, nxrange as rge
from ..meta import nxdescriptors as nxd
from ..data import datalogs as dtl
from ..io.store import Store
from . import logger as LOGGER
from .exceptions import InputError

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class TaskInput(nxd.NxAttribute):
    __registry__ = '__inputs__'

    def __init__(self, title, relation=None):
        self.title = title
        self.relation = relation
        super().__init__()

    def __validate__(self, task, value):
        if self.relation is None:
            type_ = dtl.DataSequence
        else:
            target = getattr(task.entity, self.relation)
            if isinstance(target, Sequence):
                type_ = dtl.DataLog
            else:
                type_ = dtl.DataSequence
        if not isinstance(value, type_):
            raise TypeError()
        if not isinstance(value.schema, type(self.title.schema)):
            raise ValueError()
        return True

    def __set__(self, task, value):
        try:
            super().__set__(task, value)
        except:
            if task.logger is None:
                raise
            else:
                msg = f"Cannot set input {self.name} for {task}."
                self.logger.exception(msg, exc_info=True)

    def fetch(self, input_store, entity, dt_range):
        if input_store is None:
            msg = "An input store must be specified to retrieve task inputs"
            raise TypeError(msg)
        elif isinstance(input_store, str):
            input_store = Store[input_store]
        tract = input_store[self.title]
        try:
            if self.relation is None:
                id_range = entity.id
            else:
                target = getattr(entity, self.relation)
                if isinstance(target, Sequence):
                    id_range = [e.id for e in target]
                else:
                    id_range = target.it
        except AttributeError:
            msg = "Improper entity type or query specification."
            raise InputError(msg)
        query = tract.query(id=id_range, datetime=dt_range)
        return query.data()

    def __eq__(self, other):
        if not isinstance(other, TaskInput):
            return False
        return self.title == self.title and self.relation == self.relation

    def __hash__(self):
        return hash((self.title, self.relation))


class TaskType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        TaskInput.collect(cls, namespace)


class Task(metaclass=TaskType):
    __etype__ = None

    def __init__(self, entity, dt_range=None, input_store=None,
                 output_store=None, logger=LOGGER, **inputs):
        self.entity = entity
        self.dt_range = rge.time_range(dt_range)
        self.input_store = input_store
        self.output_store = output_store
        self.logger = logger
        for k, v in inputs.items():
            setattr(self, k, v)

    @xprops.cachedproperty
    def output(self):
        return None

    def retrieve_inputs(self):
        for name, desc in self.__inputs__.items():
            if getattr(self, name) is None:
                setattr(self, name, desc.fetch(self.input_store,
                                               self.entity,
                                               self.dt_range))

    @abc.abstractmethod
    def __function__(self):
        """Type-specific function."""
        return None

    def __call__(self, plot=False):
        """Run interface."""
        self.retrieve_inputs()
        try:
            output = self.__function__()
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
