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
import time
import types

import pandas as pd

from ..utilities import nxrange as rge, xprops, functions as fun
from ..meta import nxdescriptors as nxd
from ..data import datalogs as dtl
from ..io.store import Store, Title
from ..io import redis as nxr
from .exceptions import OperationalError
from . import logger as LOGGER

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class TaskDescriptor(nxd.NxAttribute):
    shortname = None

    def __init__(self, title, relation=None):
        if isinstance(title, str):
            title = Title[title]
        self.title = title
        self.relation = relation
        super().__init__()

    @property
    def archetype(self):
        superclasses = set(self._cls.mro())
        archetypes = {InsertTask, TransformTask, UpdateTask}
        intersection = superclasses & archetypes
        try:
            return intersection.pop()
        except KeyError:
            return TransformTask

    def __validate__(self, task, value):
        if value is None:
            return True
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
            msg = f"Cannot set {self.shortname} {self.name} for {task}."
            task.logger.exception(msg)
            raise OperationalError()

    def __eq__(self, other):
        return self.title == other.title and self.relation == other.relation

    def __hash__(self):
        return hash((self.title, self.relation))


class TaskInput(TaskDescriptor):
    shortname = 'input'
    __registry__ = '__inputs__'

    def fetch(self, entity, dt_range, store=None, nxpipe=None,
              logger=LOGGER):
        cls = self.archetype
        return cls.fetch_input(self.title, self.relation,
                               entity, dt_range, store=store, nxpipe=nxpipe,
                               logger=logger)


class TaskOutput(TaskDescriptor):
    shortname = 'output'
    __registry__ = '__outputs__'

    def __init__(self, title):
        super().__init__(title)

    def fetch(self, entity, dt_range, store=None, nxpipe=None, logger=LOGGER):
        cls = self.archetype
        return cls.fetch_output(self.title, entity, dt_range,
                                store=store, nxpipe=nxpipe,
                                logger=logger)

    def merge(self, extant, output, logger=LOGGER):
        cls = self.archetype
        return cls.merge(extant, output, logger=logger)

    def store(self, entity, sequence, store=None, nxpipe=None,
              logger=LOGGER, **metadata):
        cls = self.archetype
        return cls.store_output(self.title, entity, sequence,
                                store=store, nxpipe=nxpipe, logger=logger,
                                **metadata)


class TaskType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        TaskInput.collect(cls, namespace)
        TaskOutput.collect(cls, namespace)


class Task(metaclass=TaskType):
    __etype__ = None
    __input_store__ = None  # Default input store
    __output_store__ = None  # Default output store

    def __init__(self, entity, dt_range=None, input_store=None,
                 output_store=None, logger=LOGGER, stream_mode=False,
                 **inputs):
        self.logger = logger
        self.entity = entity
        try:
            assert isinstance(entity, self.__etype__)
        except AssertionError:
            msg = f"Improper object {entity} passed to " +\
                  f"{type(self).__name__} (expects {self.__etype__})."
            self.logger.exception(msg)
            raise OperationalError
        self.stream_mode = stream_mode
        if stream_mode:
            self.dt_range = rge.time_range((None, None))
        else:
            self.dt_range = rge.time_range(dt_range)
        self.input_store = input_store or self.__input_store__
        self.output_store = output_store or self.__output_store__
        for k, v in inputs.items():
            setattr(self, k, v)

    @classmethod
    def setup_streaming(cls, entity, input_store=None, output_store=None,
                        when=None, logger=LOGGER):
        input_store = input_store or cls.__input_store__
        output_store = output_store or cls.__output_store__
        if isinstance(input_store, str):
            input_store = Store[input_store]
        if isinstance(output_store, str):
            output_store = Store[output_store]
        if isinstance(input_store, nxr.RedisStore):
            in_pipe = nxr.NxPipe(input_store)
        else:
            in_pipe = None
        if isinstance(output_store, nxr.RedisStore):
            if output_store == input_store:
                out_pipe = in_pipe
            else:
                out_pipe = nxr.NxPipe(output_store)
        else:
            out_pipe = None
        msg = f"Setting up {cls.__name__} Task streaming for {entity}."
        LOGGER.debug(msg)
        for name, desc in cls.__outputs__.items():
            tract = output_store[desc.title]
            if isinstance(tract, nxr.RedisScroll):
                tract.setup(entity.store_id, _nxpipe=out_pipe, when=when)
                for iname, idesc in cls.__inputs__.items():
                    itract = input_store[idesc.title]
                    if isinstance(itract, nxr.RedisScroll):
                        itract.setup_subscriber(tract, entity.store_id,
                                                _nxpipe=in_pipe)
        if in_pipe is not None:
            in_pipe.execute()
        if out_pipe is not None and out_pipe != in_pipe:
            out_pipe.execute()

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

    @classmethod
    def fetch_input(cls, title, relation, entity, dt_range,
                    store=None, nxpipe=None, logger=LOGGER):
        if nxpipe is None:
            if store is None:
                msg = "A store must be specified to retrieve task data"
                logger.exception(msg)
                raise OperationalError
        else:
            store = nxpipe.store
        tract = store[title]
        try:
            if relation is None:
                id_range = entity.store_id
            else:
                target = getattr(entity, relation)
                if isinstance(target, Sequence):
                    id_range = [e.store_id for e in target]
                else:
                    id_range = target.store_id
        except AttributeError:
            msg = "Improper entity type or query specification."
            logger.exception(msg)
            raise OperationalError
        if dt_range is None:
            if nxpipe:
                nxpipe.query(tract, id=id_range)
                return
            else:
                query = tract.query(id=id_range)
                return query.data()
        else:
            if nxpipe:
                nxpipe.query(tract, id=id_range, datetime=dt_range)
                return
            else:
                query = tract.query(id=id_range, datetime=dt_range)
                return query.data()

    @classmethod
    def fetch_output(cls, title, entity, dt_range, store=None, nxpipe=None,
                     logger=LOGGER):
        return cls.fetch_input(title, None, entity, dt_range,
                               store=store, nxpipe=nxpipe, logger=logger)

    def retrieve_inputs(self):
        start = time.time()
        if isinstance(self.input_store, nxr.RedisStore):
            in_pipe = nxr.NxPipe(self.input_store)
        else:
            in_pipe = None
        inputs = []
        outputs = []
        fetch_count = 0
        for name, desc in self.__inputs__.items():
            if getattr(self, name) is None:
                fetch_count += 1
                rval = desc.fetch(self.entity, self.dt_range,
                                  self.input_store, in_pipe,
                                  logger=self.logger)
                if rval is not None:
                    setattr(self, name, rval)
                else:
                    inputs.append(name)
        if self.stream_mode:
            if isinstance(self.output_store, nxr.RedisStore):
                if self.output_store == self.input_store:
                    out_pipe = in_pipe
                else:
                    out_pipe = nxr.NxPipe(self.output_store)
            else:
                out_pipe = None
            for name, desc in self.__outputs__.items():
                if getattr(self, name) is None:
                    fetch_count += 1
                    rval = desc.fetch(self.entity, self.dt_range,
                                      self.output_store, out_pipe,
                                      logger=self.logger)
                    if rval is not None:
                        setattr(self, name, rval)
                    else:
                        outputs.append(name)
        else:
            out_pipe = None
        if in_pipe is not None:
            if in_pipe == out_pipe:
                for name, val in zip(inputs + outputs, in_pipe.execute()):
                    setattr(self, name, val)
                return
            else:
                for name, val in zip(inputs, in_pipe.execute()):
                    setattr(self, name, val)
        if out_pipe is not None:
            for name, val in zip(outputs, out_pipe.execute()):
                setattr(self, name, val)
        stop = time.time()
        runtime_ms = round(1e3 * (stop - start))
        msg = f"{self} retrieved {fetch_count} data inputs in " + \
              f"{runtime_ms:,} milliseconds."
        self.logger.debug(msg)

    @classmethod
    def merge(cls, original, new, logger=LOGGER):
        """Merges new sequence with original sequence."""
        if pd.isna(original.certification):
            return new
        divider = max(original.certification, new.dt_range.lower)
        left = original[None:divider]
        right = new[divider:None]
        data = pd.concat((left.data, right.data))
        id_range = original.id_range
        dt_range = rge.MultiTimeInterval([original.dt_range,
                                          new.dt_range]).compact
        if new.certification >= original.certification:
            certification = new.certification
        else:
            certification = original.certification
        consumption = original.consumption
        s = dtl.DataSequence(data, schema=original.schema,
                             id_range=id_range, dt_range=dt_range,
                             certification=certification,
                             consumption=consumption)
        return s

    @classmethod
    def store_output(cls, title, entity, sequence, store=None, nxpipe=None,
                     logger=LOGGER, **metadata):
        if nxpipe is None:
            if store is None:
                msg = "An output store must be specified to write task outputs"
                logger.exception(msg)
                raise OperationalError
        else:
            store = nxpipe.store
        try:
            assert sequence.id_range == entity.store_id
        except AssertionError:
            msg = f"{sequence}'s id does not match {entity}."
            logger.exception(msg)
            raise OperationalError
        tract = store[title]
        old_certificate = metadata.get('certification', None)
        return tract.write(sequence, old_certificate=old_certificate,
                           _nxpipe=nxpipe)

    def store_outputs(self, nxpipes, **metadata):
        start = time.time()
        if isinstance(self.input_store, nxr.RedisStore):
            if self.input_store in nxpipes:
                in_pipe = nxpipes[self.input_store]
            else:
                in_pipe = nxr.NxPipe(self.input_store)
        else:
            in_pipe = None
        if isinstance(self.output_store, nxr.RedisStore):
            if self.output_store in nxpipes:
                out_pipe = nxpipes[self.output_store]
            elif self.output_store == self.input_store:
                out_pipe = in_pipe
            else:
                out_pipe = nxr.NxPipe(self.output_store)
        else:
            out_pipe = None
        store_count = 0
        for name, desc in self.__outputs__.items():
            out_tract = self.output_store[desc.title]
            sequence = getattr(self, name)
            certificate = sequence.certification
            if in_pipe is not None:
                for iname, idesc in self.__inputs__.items():
                    in_tract = self.input_store[idesc.title]
                    if isinstance(in_tract, nxr.PipelineScroll):
                        in_tract.update_subscriber(out_tract,
                                                   self.entity.store_id,
                                                   certificate,
                                                   _nxpipe=in_pipe)
            meta = metadata[name]
            desc.store(self.entity, sequence, store=self.output_store,
                       nxpipe=out_pipe, **meta)
            store_count += 1
        if in_pipe is not None and self.input_store not in nxpipes:
            in_pipe.execute()
        if out_pipe not in (None, in_pipe):
            if self.output_store not in nxpipes:
                out_pipe.execute()
        stop = time.time()
        runtime_ms = round(1e3 * (stop - start))
        if not self.stream_mode:
            msg = f"{self} stored {store_count} data sequences in " + \
                  f"{runtime_ms:,} milliseconds."
            self.logger.debug(msg)

    @abc.abstractmethod
    def __function__(self):
        """Type-specific function."""
        return None

    def __call__(self, nxpipes=None, plot=False):
        """Run interface."""
        start = time.time()
        if nxpipes is None:
            nxpipes = {}
        self.retrieve_inputs()
        if self.stream_mode:
            metadata = {name: getattr(self, name).metadata
                        for name in self.__outputs__}
        try:
            outputs = self.__function__()
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for (name, desc), output in zip(self.__outputs__.items(), outputs):
                extant = getattr(self, name, None)
                if extant is None:
                    setattr(self, name, output)
                else:
                    merged = desc.merge(extant, output)
                    setattr(self, name, merged)
        except:
            msg = f"{type(self).__name__} task error on {self.inputs}."
            self.logger.exception(msg, exc_info=True)
            raise OperationalError
        if plot:
            self.plot()
        if self.stream_mode:
            self.store_outputs(nxpipes, **metadata)
        stop = time.time()
        runtime_ms = round(1e3 * (stop - start))
        if not self.stream_mode:
            msg = f"{self} completed in {runtime_ms:,} milliseconds."
            self.logger.debug(msg)
        return outputs

    def plot(self, **kwargs):
        """Optional operator plot."""
        pass

    def __repr__(self):
        return fun.iformat('entity')(self)


class InsertTaskType(TaskType):
    registry = {}

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if cls.target.title is not None:
            cls.registry[cls.target.title.name] = cls

    def __getitem__(self, key):
        if isinstance(key, Title):
            key = key.name
        return self.registry.__getitem__(key)


class InsertTask(Task, metaclass=InsertTaskType):
    """A task dedicated to appending records."""
    target = TaskOutput(None)  # Replace None with meaningful Title
    max_latency = '5m'  # Heuristic watermark
    __output_store__ = 'pipeline'

    def __init__(self, entity, *records, output_store=None, logger=LOGGER,
                 stream_mode=False, when=None, max_latency=None):
        super().__init__(entity, output_store=output_store, logger=logger,
                         stream_mode=stream_mode)
        self.inputs = records
        if when is None:
            self.when = pd.Timestamp.utcnow()
        else:
            self.when = pd.to_datetime(when, utc=True)
        max_latency = fun.get(max_latency, self.max_latency)
        self.max_latency = pd.Timedelta(max_latency)

    def __function__(self):
        schema = type(self).target.title.schema
        lower = min([r.datetime for r in self.inputs] + [pd.NaT])
        upper = max([r.datetime for r in self.inputs] + [pd.NaT])
        if self.stream_mode:
            upper = max(self.when, upper)
            lower = min(upper, lower)
        certificate = upper - self.max_latency
        return dtl.DataSequence.from_records(self.inputs, schema=schema,
                                             id_range=self.entity.store_id,
                                             dt_range=(lower, upper),
                                             certification=certificate)


def insert_task_factory(title, etype, max_latency='5m'):
        task_name = f"Insert_{title.name}"

        def exec_body(ns):
            ns.update(target=TaskOutput(title),
                      __etype__=etype,
                      max_latency=max_latency)
            return ns

        return types.new_class(task_name, (InsertTask,), exec_body=exec_body)


class TransformTask(Task):
    """A task dedicated to transforming records."""
    __input_store__ = 'pipeline'
    __output_store__ = 'pipeline'


class UpdateTask(Task):
    """A task that targets the application buffer."""
    __input_store__ = 'pipeline'
    __output_store__ = 'buffer'

    @classmethod
    def fetch_output(cls, title, entity, dt_range, store=None, nxpipe=None,
                     logger=LOGGER):
        if nxpipe is None:
            if store is None:
                msg = "A store must be specified to retrieve task data"
                logger.exception(msg)
                raise OperationalError
        else:
            store = nxpipe.store
        tract = store[title]
        try:
            id_range = entity.store_id
        except AttributeError:
            msg = "Improper entity type or query specification."
            logger.exception(msg)
            raise OperationalError
        return tract.mock(id=id_range, _nxpipe=nxpipe)

    @classmethod
    def merge(cls, original, new, logger=LOGGER):
        """Merges new sequence with original metadata."""
        return new

    @classmethod
    def store_output(cls, title, entity, sequence, store=None, nxpipe=None,
                     logger=LOGGER, **metadata):
        if nxpipe is None:
            if store is None:
                msg = "An output store must be specified to write task outputs"
                raise TypeError(msg)
        else:
            store = nxpipe.store
        try:
            assert sequence.id_range == entity.store_id
        except AssertionError:
            msg = f"{sequence}'s id does not match {entity}."
            logger.exception(msg)
            raise OperationalError
        tract = store[title]
        return tract.update(entity.store_id, sequence, old_metadata=metadata,
                            _nxpipe=nxpipe)

    def __function__(self):
        return tuple(getattr(self, name) for name in self.__inputs__)
