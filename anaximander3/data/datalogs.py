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
from collections import Sequence, OrderedDict

import pandas as pd

from ..utilities import xprops, nxrange as rge
from ..utilities.jsonmixin import jsonio
from .exceptions import ConformityError
from .dataobject import DataObjectType, DataObject
from .records import Record
from . import nxschema as sch


__all__ = []

# =============================================================================
# Archetypical log classes
# =============================================================================


@jsonio
class DataLogsBase(DataObject, Sequence):
    """Base class for all data logs."""
    __schema__ = None  # placeholder for specialized type parameter
    __id_range__ = None  # placeholder for expected id range type
    __dt_range__ = None  # placeholder for expected time range type

    def __init__(self, data, *, schema=None, id_range=None, dt_range=None,
                 cast=True, validate=False, force=False, **metadata):
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
        metadata.update({'id_range': self.id_range.serializable,
                         'dt_range': self.dt_range.serializable})
        super().__init__(data, schema=schema, cast=cast, validate=validate,
                         **metadata)

    @property
    def id_range(self):
        return self._id_range

    @property
    def dt_range(self):
        return self._dt_range

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        if self.empty:
            return self._conform(self._data)
        return self._data.reset_index()

    def _conform(self, data, force=False):
        """Primitive for cast, returning a non-indexed dataframe."""
        df = pd.DataFrame(data).reset_index()
        missing_columns = []
        mistyped_columns = []
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
                    df[name] = col.dcast(df[name], force=force)
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
        return df[list(self.columns)]

    def cast(self, data, force=False):
        """Casts supplied dataframe-like object to the object's schema.

        data must be a valid input to pandas.DataFrame.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a DataFrame
        * raises ConformityError if the data misses schema columns;
        extra columns are simply removed.
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary.

        The force flag will silently handle incorrect inputs, such as
        unreadable datetime, and treat them as missing values.
        """
        df = self._conform(data, force=force)
        df.set_index(self._index_columns, drop=True, inplace=True)
        return df.sort_index()

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

    @property
    def empty(self):
        return self.data.empty

    def _metaslice(self, id_slice=None, dt_slice=None):
        """Primitive for __getitem__, providing metadata."""
        if id_slice is None:
            id_range = self.id_range
        else:
            id_range = rge.cat_range(id_slice, sliced=self.id_range)
            id_range &= self.id_range
        if dt_slice is None:
            dt_range = self.dt_range
        else:
            dt_range = rge.time_range(dt_slice) & self.dt_range
        metadata = self.metadata.copy()
        metadata['id_range'] = id_range
        metadata['dt_range'] = dt_range
        return metadata

    def _dataslice(self, key, id_range, dt_range):
        """Data slicer."""
        return self.data.loc[key, :]

    def _slice(self, data, metadata):
        """Returns a dataobject slice from data and metadata."""
        id_range = metadata['id_range']
        dt_range = metadata['dt_range']
        archetype_ = archetype(id_range, dt_range)
        if isinstance(data, pd.DataFrame):
            data.reset_index(inplace=True)
            if archetype_ is Record:
                data = data.iloc[0]
        elif isinstance(data, pd.Series):
            data['id'] = id_range.level
            data['datetime'] = dt_range.position
        else:  # Single index of single column
            col_name = list(self.columns)[-1]
            data = {'id': id_range.level,
                    'datetime': dt_range.position,
                    col_name: data}
        return archetype_(data, schema=self.schema, **metadata)

    def __len__(self):
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, key):
        """Slices the data per index properties."""
        return NotImplemented

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self._data.equals(other._data) and \
            self.metadata == other.metadata

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

    def to_dict(self, **kwargs):
        return {'data': self.tabulated.to_dict(),
                'schema': self.schema.to_dict(),
                'metadata': self.metadata}

    @classmethod
    def from_dict(cls, dict_, **kwargs):
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
        validate = kwargs.get('validate', False)
        return cls(data, schema=schema, id_range=id_range,
                   dt_range=dt_range, validate=validate, **metadata)

    @classmethod
    def from_records(cls, records, id_range=None, dt_range=None,
                     cast=True, validate=False, **metadata):
        try:
            schema_set = set(r.schema for r in records)
            assert len(schema_set) == 1
            schema = schema_set.pop()
        except AssertionError:
            msg = "All records must share the same schema."
            raise ConformityError(msg)
        rows = [r.tabulated for r in records]
        data = pd.DataFrame.from_records(rows)
        return cls(data, schema=schema, id_range=id_range, dt_range=dt_range,
                   cast=cast, validate=validate, **metadata)

    def __repr__(self):
        return f"<{type(self).__name__} id_range:{str(self.id_range)} " + \
               f"dt_range:{str(self.dt_range)}>"


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

    @property
    def id(self):
        return self._data.index.get_level_values(0).copy()

    @property
    def datetime(self):
        return self._data.index.get_level_values(1).copy()

    def _xdataslice(self, key, id_range, dt_range):
        if isinstance(id_range, rge.Level):
            id = id_range.level
            if isinstance(dt_range, rge.TimeInterval):
                lower = self._previous(id, dt_range.lower)
                upper = dt_range.upper
                key = (id, slice(lower, upper))
                return self.data.loc[key, :]
            elif isinstance(dt_range, rge.TimeSingleton):
                position = self._previous(id, dt_range.position)
                if position is None:
                    raise KeyError
                else:
                    key = (id, position)
                    return self.data.loc[key, :]
        elif isinstance(id_range, rge.Levels):
            if isinstance(dt_range, rge.TimeInterval):
                lowers = {id: self._previous(id, dt_range.lower)
                          for id in id_range}
                upper = dt_range.upper
                dataframes = [self.data.loc[(id, slice(lower, upper)), :]
                              for id, lower in lowers.items()]
                return pd.concat(dataframes)
            elif isinstance(dt_range, rge.TimeSingleton):
                positions = {id: self._previous(id, dt_range.position)
                             for id in id_range}
                locs = [(k, v) for k, v in positions.items() if v is not None]
                return self.data.loc[locs, :]

    def _dataslice(self, key, id_range, dt_range):
        if self.schema.xindex:
            return self._xdataslice(key, id_range, dt_range)
        return self.data.loc[key, :]

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.index[key]
        if isinstance(key, tuple):
            id_slice, dt_slice = key
        elif isinstance(key, slice):
            id_slice, dt_slice = None, key
            key = (slice(None), dt_slice)
        else:
            id_slice, dt_slice = key, None
        metadata = self._metaslice(id_slice, dt_slice)
        id_range = metadata['id_range']
        dt_range = metadata['dt_range']
        data = self._dataslice(key, id_range, dt_range)
        return self._slice(data, metadata)

    def _previous(self, id, dt):
        """Returns immediately previous datetime in index for id, or None."""
        mindex = self._data.index
        dt_index = mindex[mindex.get_loc(id)].get_level_values(1)
        if dt in dt_index:
            return dt
        dt_ix = dt_index.searchsorted(dt)
        if dt_ix > 0:
            return dt_index[dt_ix - 1]
        else:
            return None


class DataSequence(DataLogsBase, metaclass=SequenceType):
    """A single-id dataframe container, indexed by datetime."""
    __schema__ = sch.LogsSchema
    __id_range__ = rge.Level
    __dt_range__ = rge.TimeInterval
    _index_columns = ['datetime']

    @property
    def id(self):
        return self.id_range.level

    @property
    def datetime(self):
        return self._data.index.copy()

    @property
    def columns(self):
        cols = OrderedDict(self.schema)
        del cols['id']
        return cols

    @property
    def idx(self):
        idx_ = list(zip([self.id_range.level], self.index))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        df = self._data.reset_index()
        df.insert(0, 'id', self.id_range.level)
        return df

    def _xdataslice(self, key, dt_range):
        if isinstance(dt_range, rge.TimeInterval):
            lower = self._previous(dt_range.lower)
            upper = dt_range.upper
            key = slice(lower, upper)
        elif isinstance(dt_range, rge.TimeSingleton):
            position = self._previous(dt_range.position)
            if position is None:
                raise KeyError
            else:
                key = position
        return self.data.loc[key]

    def _dataslice(self, key, dt_range):
        if self.schema.xindex:
            return self._xdataslice(key, dt_range)
        return self.data.loc[key]

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.index[key]
        dt_slice = key
        metadata = self._metaslice(dt_slice=dt_slice)
        dt_range = metadata['dt_range']
        data = self._dataslice(key, dt_range)
        return self._slice(data, metadata)

    def _previous(self, dt):
        """Returns the immediately previous datetime in the index, or None."""
        index = self._data.index
        if dt in index:
            return dt
        dt_ix = index.searchsorted(dt)
        if dt_ix > 0:
            return index[dt_ix - 1]
        else:
            return None


class DataArray(DataLogsBase, metaclass=ArrayType):
    """A single datetime dataframe container, indexed by id."""
    __schema__ = sch.LogsSchema
    __id_range__ = rge.Levels
    __dt_range__ = rge.TimeSingleton
    _index_columns = ['id']

    @property
    def id(self):
        return self._data.index.copy()

    @property
    def datetime(self):
        return self.dt_range.position

    @property
    def columns(self):
        cols = OrderedDict(self.schema)
        del cols['datetime']
        return cols

    @property
    def idx(self):
        idx_ = list(zip(self.index, [self.dt_range.position]))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        df = self._data.reset_index()
        df.insert(1, 'datetime', self.dt_range.position)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        return df

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self.index[key]
        id_slice = key
        metadata = self._metaslice(id_slice=id_slice)
        data = self.data.loc[key]
        return self._slice(data, metadata)


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


class EventLog(DataLog):
    __schema__ = sch.EventLogsSchema


class StateLog(DataLog):
    __schema__ = sch.StateLogsSchema


class SessionLog(DataLog):
    __schema__ = sch.SessionLogsSchema

    @property
    def start(self):
        """Start times of sessions."""
        return pd.Series(self.datetime, self.index)

    @property
    def stop(self):
        """Stop times of sessions."""
        return self.start + self.duration


class PeriodLog(DataLog):
    __schema__ = sch.PeriodLogsSchema


class SampleSequence(DataSequence):
    __schema__ = sch.SampleLogsSchema


class EventSequence(DataSequence):
    __schema__ = sch.EventLogsSchema


class StateSequence(DataSequence):
    __schema__ = sch.StateLogsSchema


class SessionSequence(DataSequence):
    __schema__ = sch.SessionLogsSchema

    @property
    def start(self):
        """Start times of sessions."""
        return pd.Series(self.datetime, self.index)

    @property
    def stop(self):
        """Stop times of sessions."""
        return self.start + self.duration


class PeriodSequence(DataSequence):
    __schema__ = sch.PeriodLogsSchema


class SampleArray(DataArray):
    __schema__ = sch.SampleLogsSchema


class EventArray(DataArray):
    __schema__ = sch.EventLogsSchema


class StateArray(DataArray):
    __schema__ = sch.StateLogsSchema


class SessionArray(DataArray):
    __schema__ = sch.SessionLogsSchema

    @property
    def start(self):
        """Start times of sessions."""
        return pd.Series(self.datetime, self.index)

    @property
    def stop(self):
        """Stop times of sessions."""
        return self.start + self.duration


class PeriodArray(DataArray):
    __schema__ = sch.PeriodLogsSchema
