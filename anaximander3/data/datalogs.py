#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines datalogs, containers of columnar time series data.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import OrderedDict
import json

import pandas as pd

from ..utilities import xprops, nxrange as rge, jsonmixin
from .exceptions import ConformityError
from .base import DataObjectType
from .records import Record
from . import nxschema as sch


__all__ = []

# =============================================================================
# Archetypical log classes
# =============================================================================


class DataLogsBase(jsonmixin.JsonMixin):
    """Base class for all data logs."""
    __schema__ = None  # placeholder for specialized type parameter
    __id_range__ = None  # placeholder for expected id range type
    __dt_range__ = None  # placeholder for expected time range type

    #TODO: this should inspect the schema's inheritance to provide
    # the highest-registered schemaed class
    def __new__(cls, data, *, schema=None, id_range=None, dt_range=None,
                cast=True, validate=False, **metadata):
        if schema is not None:
            try:
                type_ = cls[schema]
            except KeyError:
                pass
            else:
                return type_(data, schema=schema, id_range=id_range,
                             dt_range=dt_range, cast=cast,  **metadata)
        return super().__new__(cls)

    def __init__(self, data, *, schema=None, id_range=None, dt_range=None,
                 cast=True, validate=False, **metadata):
        if schema is not None:
            if isinstance(schema, type):
                self.schema = schema()
            else:
                self.schema = schema
        else:
            self.schema = self.__schema__()
        try:
            assert isinstance(self.schema, self.__schema__)
        except AssertionError:
            msg = f"Improper schema supplied to {type(self).__name__}. " + \
                  f"It must be of type {self.__schema__.__name__}"
            raise ConformityError(msg)
        if cast:
            self._data = self.cast(data)
        else:
            self._data = data
        self._id_range = rge.cat_range(id_range)
        self._dt_range = rge.time_range(dt_range)
        try:
            assert isinstance(self._id_range, self.__id_range__)
        except AssertionError:
            msg = f"{type(self).__name__} expects a " + \
                  f"{self.__id_range__.__name__} instance that could not " + \
                  f"be cast from {id_range}"
            raise ValueError(msg)
        try:
            assert isinstance(self._dt_range, self.__dt_range__)
        except AssertionError:
            msg = f"{type(self).__name__} expects a " + \
                  f"{self.__dt_range__.__name__} instance that could not " + \
                  f"be cast from {dt_range}"
            raise ValueError(msg)
        self.metadata = metadata
        self.metadata.update({'id_range': self.id_range.serializable,
                              'dt_range': self.dt_range.serializable})
        if validate:
            if not self.validate():
                msg = "Validation failed with the following errors: " + \
                      "{self.errors}"
                raise ConformityError(msg)

    @property
    def data(self):
        return self._data.copy()

    @property
    def id_range(self):
        return self._id_range

    @property
    def dt_range(self):
        return self._dt_range

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        return self._data.reset_index()

    @property
    def columns(self):
        return OrderedDict(self.schema)

    def cast(self, data):
        """Casts supplied dataframe-like object to the object's schema.

        data must be a valid input to pandas.DataFrame.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a DataFrame
        * raises ConformityError if the data misses schema columns;
        extra columns are simply removed.
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary.
        """
        df = pd.DataFrame(data).reset_index()
        missing_columns = []
        mistyped_columns = []
        if df.empty:
            dataframe = pd.DataFrame(columns=self.columns)
            dataframe.dtypes = [c.dtype for c in self.columns.values()]
        else:
            for name, col in self.columns.items():
                if name not in df:
                    if col.default is not None:
                        df[name] = col.default
                    else:
                        missing_columns.append(name)
                        continue
                if col.missing is not None:
                    df[name] = df[name].fillna(col.missing)
                try:
                    assert df.dtypes[name] == col.dtype
                except (AssertionError, TypeError):
                    try:
                        df[name] = col.dcast(df[name])
                    except (ValueError, TypeError):
                        mistyped_columns.append(name)
            if missing_columns:
                msg = f"Data is missing schema columns {missing_columns}."
                raise ConformityError(msg)
            if mistyped_columns:
                dtypes = [c.dtype for c in
                          [self.columns[n] for n in mistyped_columns]]
                msg = f"Could not cast {mistyped_columns} to required " + \
                      f"dtypes {dtypes}"
                raise ConformityError(msg)
            dataframe = df[list(self.columns)]
        dataframe.set_index(self._index_columns, drop=True, inplace=True)
        return dataframe.sort_index()

    @property
    def index(self):
        """Returns the object's index."""
        return self._data.index.copy()

    @abc.abstractproperty
    def idx(self):
        """Returns a series of indexing tuples."""
        return pd.Series([()], index=self.index)

    @property
    def key(self):
        """Returns a series of row keys, indexed by self's index."""
        return self.idx.apply(self.schema.rowkey).rename('key')

    @abc.abstractmethod
    def __getitem__(self, key):
        """Slices the data per index properties."""
        return NotImplemented

    @xprops.cachedproperty
    def errors(self):
        return self._validate()

    def _validate(self):
        """Runs column validators, returns a dataframe of found errors."""
        df = self.tabulated
        errors = df[['id', 'datetime']]
        error_columns = []
        for name, col in self.columns.items():
            for i, v in enumerate(col.validators):
                valid = df.apply(lambda r: v(r, col, r[name]), axis=1)
                if not all(valid == True):
                    cnm = name + "_errors" + (("_" + str(i)) if i > 0 else '')
                    error_columns.append(cnm)
                    errors[cnm] = valid
        if not error_columns:
            return pd.DataFrame()
        error_row = lambda r: any(r != True)
        errors = errors[errors[error_columns].apply(error_row, axis=1)]
        errors.set_index(['id', 'datetime'], drop=True, inplace=True)
        errors = errors.sort_index()
        self._errors = errors
        return errors

    def validate(self):
        """Returns True or False whether the data validates or not."""
        return self._validate().empty

    def to_dict(self):
        return {'data': json.loads(self.tabulated.to_json(date_format='iso')),
                'schema': self.schema.to_dict(),
                'metadata': self.metadata}

    @classmethod
    def from_dict(cls, dict_):
        try:
            data_ = dict_['data']
            schema_ = dict_['schema']
            metadata = dict_['metadata']
        except KeyError:
            msg = f"Invalid mapping."
            raise ValueError(msg)
        data = pd.DataFrame.from_dict(data_)
        schema = sch.Schema.from_dict(schema_)
        id_range = rge.cat_range(metadata.pop('id_range', None))
        dt_range = rge.time_range(metadata.pop('dt_range', None))
        cls = archetype(id_range, dt_range)
        return cls(data, schema=schema, id_range=id_range,
                   dt_range=dt_range, **metadata)


class LogType(DataObjectType):
    _registry = dict()


class SequenceType(DataObjectType):
    _registry = dict()


class ArrayType(DataObjectType):
    _registry = dict()


class DataLog(DataLogsBase, metaclass=LogType):
    """A doubly-index log -multiple ids and datetimes."""
    __schema__ = sch.LogsSchema
    __id_range__ = rge.Levels
    __dt_range__ = rge.TimeInterval
    _index_columns = ['id', 'datetime']

    @property
    def idx(self):
        return pd.DataFrame({'idx': list(self.index)}, index=self.index).idx

    def __getitem__(self, key):
        if isinstance(key, tuple):
            id_key, dt_key = key
        else:
            id_key, dt_key = key, self.dt_range
        range_id = isinstance(id_key, (list, slice))
        # XXX: must address the case where dt_key is a time interval
        range_dt = isinstance(dt_key, (list, slice))
        data = pd.DataFrame(self.data[key])
        if range_id and range_dt:
            type_ = type(self)
        elif range_id:
            type_ = DataArray[self.schema]
        elif range_dt:
            type_ = DataSequence[self.schema]
        # TODO: Record case
        else:
            return data
        return type_(data, id_range=id_key, dt_range=dt_key,
                     cast=False, schema=self.schema, **self.metadata)


class DataSequence(DataLogsBase, metaclass=SequenceType):
    """A single-id dataframe container, indexed by datetime."""
    __schema__ = sch.LogsSchema
    __id_range__ = rge.Level
    __dt_range__ = rge.TimeInterval
    _index_columns = ['datetime']

    @property
    def id(self):
        return self.id_range

    @property
    def columns(self):
        cols = OrderedDict(self.schema)
        del cols['id']
        return cols

    @property
    def idx(self):
        idx_ = list(zip([self.id_range], self.index))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        df = self._data.reset_index()
        df.insert(0, 'id', self.id_range)
        df['id'] = df['id'].astype('category')
        return df

    def __getitem__(self, key):
        range_dt = isinstance(key, (list, slice))
        data = pd.DataFrame(self.data[key])
        if range_dt:
            type_ = type(self)
        # TODO: Record case
        else:
            return data
        return type_(data, id_range=self.id_range, dt_range=key,
                     cast=False, schema=self.schema, **self.metadata)


class DataArray(DataLogsBase, metaclass=ArrayType):
    """A single datetime dataframe container, indexed by id."""
    __schema__ = sch.LogsSchema
    __id_range__ = rge.Levels
    __dt_range__ = rge.TimeSingleton
    _index_columns = ['id']

    @property
    def datetime(self):
        return self.dt_range

    @property
    def columns(self):
        cols = OrderedDict(self.schema)
        del cols['datetime']
        return cols

    @property
    def idx(self):
        idx_ = list(zip(self.index, [self.dt_range]))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        df = self._data.reset_index()
        df.insert(1, 'datetime', self.dt_range)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        return df

    def __getitem__(self, key):
        range_id = isinstance(key, (list, slice))
        data = pd.DataFrame(self.data[key])
        if range_id:
            type_ = type(self)
        # TODO: Record case
        else:
            return data
        return type_(data, id_range=key, dt_range=self.dt_range,
                     cast=False, schema=self.schema, **self.metadata)


def archetype(id_range, dt_range):
    """Selects an archetype based on the type of ranges supplied."""
    if isinstance(id_range, rge.Level):
        if isinstance(dt_range, rge.TimeSingleton):
            return Record
        elif isinstance(dt_range, rge.TimeInterval):
            return DataSequence
    elif isinstance(id_range, rge.Levels):
        if isinstance(dt_range, rge.TimeSingleton):
            return DataArray
        elif isinstance(dt_range, rge.TimeInterval):
            return DataLog
    else:
        raise TypeError


# =============================================================================
# Practical log types
# =============================================================================


class SampleLog(DataLog):
    __schema__ = sch.SampleLogsSchema


class SoftEventLog(DataLog):
    __schema__ = sch.SoftEventLogsSchema


class HardEventLog(DataLog):
    __schema__ = sch.HardEventLogsSchema


class StateLog(DataLog):
    __schema__ = sch.StateLogsSchema


class SummaryLog(DataLog):
    __schema__ = sch.SummaryLogsSchema


class SampleSequence(DataSequence):
    __schema__ = sch.SampleLogsSchema


class SoftEventSequence(DataSequence):
    __schema__ = sch.SoftEventLogsSchema


class HardEventSequence(DataSequence):
    __schema__ = sch.HardEventLogsSchema


class StateSequence(DataSequence):
    __schema__ = sch.StateLogsSchema


class SummarySequence(DataSequence):
    __schema__ = sch.SummaryLogsSchema


class SampleArray(DataArray):
    __schema__ = sch.SampleLogsSchema


class SoftEventArray(DataArray):
    __schema__ = sch.SoftEventLogsSchema


class HardEventArray(DataArray):
    __schema__ = sch.HardEventLogsSchema


class StateArray(DataArray):
    __schema__ = sch.StateLogsSchema


class SummaryArray(DataArray):
    __schema__ = sch.SummaryLogsSchema
