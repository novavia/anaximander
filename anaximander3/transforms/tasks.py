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

import pandas as pd

from ..utilities import nxrange as rge, xprops, functions as fun
from ..meta import nxdescriptors as nxd
from ..data import datalogs as dtl
from ..io.store import Store
from ..io import redis as nxr
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
            if task.logger is None:
                raise
            else:
                msg = f"Cannot set {self.shortname} {self.name} for {task}."
                self.logger.exception(msg, exc_info=True)

#    def fetch(self, entity, nxpipe=None, store=None, dt_range=None):
#        if nxpipe is None:
#            if store is None:
#                msg = "A store must be specified to retrieve task data"
#                raise TypeError(msg)
#            elif isinstance(store, str):
#                store = Store[store]
#        else:
#            store = nxpipe.store
#        tract = store[self.title]
#        try:
#            if self.relation is None:
#                id_range = entity.id
#            else:
#                target = getattr(entity, self.relation)
#                if isinstance(target, Sequence):
#                    id_range = [e.id for e in target]
#                else:
#                    id_range = target.id
#        except AttributeError:
#            msg = "Improper entity type or query specification."
#            raise IOError(msg)
#        if dt_range is None:
#            if nxpipe:
#                nxpipe.query(tract, id=id_range)
#                return
#            else:
#                query = tract.query(id=id_range)
#                return query.data()
#        else:
#            if nxpipe:
#                nxpipe.query(tract, id=id_range, datetime=dt_range)
#                return
#            else:
#                query = tract.query(id=id_range, datetime=dt_range)
#                return query.data()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.title == other.title and self.relation == other.relation

    def __hash__(self):
        return hash((self.title, self.relation))


class TaskInput(TaskDescriptor):
    shortname = 'input'
    __registry__ = '__inputs__'

    def fetch(self, entity, dt_range, store=None, nxpipe=None):
        cls = self.archetype
        return cls.fetch_input(self.title, self.relation,
                               entity, dt_range, store=store, nxpipe=nxpipe)


class TaskOutput(TaskDescriptor):
    shortname = 'output'
    __registry__ = '__outputs__'

    def __init__(self, title):
        super().__init__(title)

    def fetch(self, entity, dt_range, store=None, nxpipe=None):
        cls = self.archetype
        return cls.fetch_output(self.title, self.relation,
                                entity, dt_range, store=store, nxpipe=nxpipe)

    def merge(self, extant, output):
        cls = self.archetype
        return cls.merge(extant, output)

    def store(self, entity, sequence, store=None, nxpipe=None, **metadata):
        cls = self.archetype
        return cls.store_output(self.title, entity, sequence,
                                store=store, nxpipe=nxpipe, **metadata)

#    def store(self, entity, sequence, nxpipe=None, output_store=None,
#              old_certificate=None):
#        if nxpipe is None:
#            if output_store is None:
#                msg = "An output store must be specified to write task outputs"
#                raise TypeError(msg)
#            elif isinstance(output_store, str):
#                output_store = Store[output_store]
#        else:
#            output_store = nxpipe.store
#        try:
#            assert sequence.id_range == entity.id
#        except AssertionError:
#            msg = f"{sequence}'s id does not match {entity}."
#            raise ValueError(msg)
#        tract = output_store[self.title]
#        return tract.write(sequence, old_certificate=old_certificate,
#                           _nxpipe=nxpipe)

#    def merge(self, original, new):
#        """Merges new sequence with original sequence."""
#        if pd.isna(original.certification):
#            return new
#        divider = max(original.certification, new.dt_range.lower)
#        left = original[None:divider]
#        right = new[divider:None]
#        data = pd.concat((left.data, right.data))
#        id_range = original.id_range
#        dt_range = rge.MultiTimeInterval([original.dt_range,
#                                          new.dt_range]).compact
#        if new.certification >= original.certification:
#            certification = new.certification
#        else:
#            certification = original.certification
#        consumption = original.consumption
#        return dtl.DataSequence(data, schema=self.title.schema,
#                                id_range=id_range, dt_range=dt_range,
#                                certification=certification,
#                                consumption=consumption)


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
        self.entity = entity
        self.stream_mode = stream_mode
        if stream_mode:
            self.dt_range = rge.time_range((None, None))
        else:
            self.dt_range = rge.time_range(dt_range)
        self.input_store = input_store or self.__input_store__
        self.output_store = output_store or self.__output_store__
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

    @classmethod
    def fetch_input(cls, title, relation, entity, dt_range,
                    store=None, nxpipe=None):
        if nxpipe is None:
            if store is None:
                msg = "A store must be specified to retrieve task data"
                raise TypeError(msg)
        else:
            store = nxpipe.store
        tract = store[title]
        try:
            if relation is None:
                id_range = entity.id
            else:
                target = getattr(entity, relation)
                if isinstance(target, Sequence):
                    id_range = [e.id for e in target]
                else:
                    id_range = target.id
        except AttributeError:
            msg = "Improper entity type or query specification."
            raise IOError(msg)
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
    def fetch_output(cls, title, relation, entity, dt_range,
                     store=None, nxpipe=None):
        return cls.fetch_input(title, relation, entity, dt_range,
                               store=store, nxpipe=nxpipe)

    def retrieve_inputs(self):
        if isinstance(self.input_store, nxr.RedisStore):
            in_pipe = nxr.NxPipe(self.input_store)
        else:
            in_pipe = None
        inputs = []
        outputs = []
        for name, desc in self.__inputs__.items():
            if getattr(self, name) is None:
                rval = desc.fetch(self.entity, self.dt_range,
                                  self.input_store, in_pipe)
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
                    rval = desc.fetch(self.entity, self.dt_range,
                                      self.output_store, out_pipe)
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

    @classmethod
    def merge(cls, original, new):
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
        return dtl.DataSequence(data, schema=original.schema,
                                id_range=id_range, dt_range=dt_range,
                                certification=certification,
                                consumption=consumption)

    @classmethod
    def store_output(cls, title, entity, sequence, store=None, nxpipe=None,
                     **metadata):
        if nxpipe is None:
            if store is None:
                msg = "An output store must be specified to write task outputs"
                raise TypeError(msg)
        else:
            store = nxpipe.store
        try:
            assert sequence.id_range == entity.id
        except AssertionError:
            msg = f"{sequence}'s id does not match {entity}."
            raise ValueError(msg)
        tract = store[title]
        old_certificate = metadata.get('old_certificate', None)
        return tract.write(sequence, old_certificate=old_certificate,
                           _nxpipe=nxpipe)

    def store_outputs(self, nxpipes, **kwargs):
        certificates = kwargs
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
        for name, desc in self.__outputs__.items():
            out_tract = self.output_store[desc.title]
            sequence = getattr(self, name)
            certificate = sequence.certification
            if in_pipe is not None:
                for iname, idesc in self.__inputs__.items():
                    in_tract = self.input_store[idesc.title]
                    in_tract.update_subscriber(out_tract,
                                               self.entity.id,
                                               certificate,
                                               _nxpipe=in_pipe)
            old_certificate = certificates[name]
            desc.store(self.entity, sequence, store=self.output_store,
                       nxpipe=out_pipe, old_certificate=old_certificate)
        if in_pipe is not None and self.input_store not in nxpipes:
            in_pipe.execute()
        if out_pipe not in (None, in_pipe):
            if self.output_store not in nxpipes:
                out_pipe.execute()

    @abc.abstractmethod
    def __function__(self):
        """Type-specific function."""
        return None

    def __call__(self, nxpipes=None, plot=False):
        """Run interface."""
        if nxpipes is None:
            nxpipes = {}
        self.retrieve_inputs()
        if self.stream_mode:
            certificates = {name: getattr(self, name).certification
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
            if self.logger is None:
                raise
            msg = f"{type(self).__name__} task error on {self.inputs}."
            self.logger.exception(msg, exc_info=True)
            return None
        if plot:
            self.plot()
        if self.stream_mode:
            self.store_outputs(nxpipes, **certificates)
        return outputs

    def plot(self, **kwargs):
        """Optional operator plot."""
        pass


class InsertTask(Task):
    """A task dedicated to appending records."""
    target = TaskOutput(None)  # Replace None with meaningful Title
    max_latency = '5m'  # Heuristic watermark
    __output_store__ = 'process'

    def __init__(self, entity, *records, output_store=None, logger=LOGGER,
                 stream_mode=False, when=None, max_latency=None):
        super().__init__(entity, output_store=output_store, logger=logger,
                         stream_mode=stream_mode)
        self.records = records
        if when is None:
            self.when = pd.Timestamp.utcnow()
        else:
            self.when = pd.to_datetime(when, utc=True)
        max_latency = fun.get(max_latency, self.max_latency)
        self.max_latency = pd.Timedelta(max_latency)

    def __function__(self):
        schema = type(self).target.title.schema
        lower = min([r.datetime for r in self.records] + [pd.NaT])
        upper = max([r.datetime for r in self.records] + [pd.NaT])
        if self.stream_mode:
            upper = max(self.when, upper)
            lower = min(upper, lower)
        certificate = upper - self.max_latency
        return dtl.DataSequence.from_records(self.records, schema=schema,
                                             id_range=self.entity.id,
                                             dt_range=(lower, upper),
                                             certification=certificate)


class TransformTask(Task):
    """A task dedicated to transforming records."""
    __input_store__ = 'process'
    __output_store__ = 'process'


class UpdateTask(Task):
    """A task that targets the application buffer."""
    __input_store__ = 'process'
    __output_store__ = 'buffer'

    @classmethod
    def fetch_output(cls, title, relation, entity, dt_range,
                     store=None, nxpipe=None):
        if nxpipe is None:
            if store is None:
                msg = "A store must be specified to retrieve task data"
                raise TypeError(msg)
        else:
            store = nxpipe.store
        tract = store[title]
        try:
            if relation is None:
                id_range = entity.id
            else:
                target = getattr(entity, relation)
                if isinstance(target, Sequence):
                    msg = "Specification is incompatible with task type."
                    raise TypeError(msg)
                else:
                    id_range = target.id
        except AttributeError:
            msg = "Improper entity type or query specification."
            raise IOError(msg)
        return tract.metadata(id=id_range, nxpipe=nxpipe)

    def __function__(self):
        return self.inputs
