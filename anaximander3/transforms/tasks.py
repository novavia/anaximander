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
from itertools import chain

from ..utilities import nxrange as rge, xprops
from ..meta import nxdescriptors as nxd
from ..data import datalogs as dtl
from ..io.store import Store
from . import logger as LOGGER

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class TaskDescriptor(nxd.NxAttribute):
    shortname = None

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
                msg = f"Cannot set {self.shortname} {self.name} for {task}."
                self.logger.exception(msg, exc_info=True)

    def fetch(self, store, entity, dt_range):
        if store is None:
            msg = "A store must be specified to retrieve task data"
            raise TypeError(msg)
        elif isinstance(store, str):
            store = Store[store]
        tract = store[self.title]
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
            raise IOError(msg)
        query = tract.query(id=id_range, datetime=dt_range)
        return query.data()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.title == other.title and self.relation == other.relation

    def __hash__(self):
        return hash((self.title, self.relation))


class TaskInput(TaskDescriptor):
    shortname = 'input'
    __registry__ = '__inputs__'


class TaskOutput(TaskDescriptor):
    shortname = 'output'
    __registry__ = '__outputs__'

    def __init__(self, title):
        super().__init__(title)

    def store(self, sequence, output_store, entity, old_certificate=None):
        if output_store is None:
            msg = "An output store must be specified to write task outputs"
            raise TypeError(msg)
        elif isinstance(output_store, str):
            output_store = Store[output_store]
        try:
            assert sequence.id_range == entity.id
        except AssertionError:
            msg = f"{sequence}'s id does not match {entity}."
            raise ValueError(msg)
        tract = output_store[self.title]
        return tract.write(sequence, old_certificate=old_certificate)


class TaskType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        TaskInput.collect(cls, namespace)
        TaskOutput.collect(cls, namespace)


class Task(metaclass=TaskType):
    __etype__ = None

    def __init__(self, entity, dt_range=None, input_store=None,
                 output_store=None, logger=LOGGER, stream_mode=False,
                 **inputs):
        self.entity = entity
        self.stream_mode = stream_mode
        if stream_mode:
            self.dt_range = rge.time_range((None, None))
        else:
            self.dt_range = rge.time_range(dt_range)
        self.input_store = input_store
        self.output_store = output_store
        self.logger = logger
        for k, v in inputs.items():
            setattr(self, k, v)

    @classmethod
    def setup_streaming(cls, entity, input_store, output_store, logger=LOGGER):
        if isinstance(input_store, str):
            input_store = Store[input_store]
        if isinstance(output_store, str):
            output_store = Store[output_store]
        for name, desc in cls.__outputs__.items():
            tract = output_store[desc.title]
            tract.setup(entity.id)
            for iname, idesc in cls.__inputs__.items():
                itract = input_store[idesc.title]
                itract.setup_subscriber(tract, entity.id)

    @xprops.settablecachedproperty
    def input_store(self):
        return None

    @input_store.setter
    def input_store(self, value):
        if isinstance(value, str):
            value = Store[value]
        setattr(self, '_input_store', value)

    @xprops.settablecachedproperty
    def output_store(self):
        return None

    @output_store.setter
    def output_store(self, value):
        if isinstance(value, str):
            value = Store[value]
        setattr(self, '_output_store', value)

    def retrieve_inputs(self):
        for name, desc in self.__inputs__.items():
            if getattr(self, name) is None:
                setattr(self, name, desc.fetch(self.input_store,
                                               self.entity,
                                               self.dt_range))
        if self.stream_mode:
            for name, desc in self.__outputs__.items():
                if getattr(self, name) is None:
                    setattr(self, name, desc.fetch(self.input_store,
                                                   self.entity))

    @abc.abstractmethod
    def __function__(self):
        """Type-specific function."""
        return None

    def __call__(self, plot=False):
        """Run interface."""
        self.retrieve_inputs()
        if self.stream_mode:
            self.retrieve_outputs()
            certificates = {name: getattr(self, name).certification
                            for name in self.__outputs__}
        try:
            outputs = self.__function__()
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for name, output in zip(self.__outputs__, outputs):
                setattr(self, name, output)
        except:
            if self.logger is None:
                raise
            msg = f"{type(self).__name__} task error on {self.inputs}."
            self.logger.exception(msg, exc_info=True)
            return None
        if plot:
            self.plot()
        if self.stream_mode:
            for name, desc in self.__outputs__.items():
                out_tract = self.output_store[desc.title]
                sequence = getattr(self, name)
                certificate = sequence.certification
                for iname, idesc in self.__inputs__.items():
                    in_tract = self.input_store[idesc.title]
                    in_tract.update_subscriber(out_tract, self.entity.id,
                                               certificate)
                old_certificate = certificates[name]
                desc.store(sequence, self.output_store, self.entity,
                           old_certificate)
        return outputs

    def plot(self, **kwargs):
        """Optional operator plot."""
        pass
