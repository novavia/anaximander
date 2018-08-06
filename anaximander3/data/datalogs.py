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
from itertools import cycle

import pandas as pd

from ..utilities import xprops, nxrange as rge, functions as fun, nxtime
from ..utilities.jsonmixin import jsonio
from .exceptions import ConformityError
from .dataobject import DataObjectType, DataObject
from .records import Record
from . import nxschema as sch, nxcolumns as cln
from . import plot


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

    def __init__(self, data=None, *, schema=None, id_range=None, dt_range=None,
                 cast=True, flex=True, validate=False, force=False,
                 certification=None, consumption=None, **metadata):
        id_range = rge.cat_range(id_range)
        dt_range = rge.time_range(dt_range)
        try:
            assert isinstance(id_range, self.__id_range__)
        except AssertionError:
            msg = f"{type(self).__name__} expects a " + \
                  f"{self.__id_range__.__name__} instance that could not " + \
                  f"be cast from {id_range}"
            raise ValueError(msg)
        try:
            assert isinstance(dt_range, self.__dt_range__)
        except AssertionError:
            msg = f"{type(self).__name__} expects a " + \
                  f"{self.__dt_range__.__name__} instance that could not " + \
                  f"be cast from {dt_range}"
            raise ValueError(msg)
        metadata.update({'id_range': id_range,
                         'dt_range': dt_range})
        certification = fun.get(certification, pd.NaT)
        metadata['certification'] = pd.to_datetime(certification, utc=True)
        consumption = fun.get(consumption, pd.NaT)
        metadata['consumption'] = pd.to_datetime(consumption, utc=True)
        super().__init__(data, schema=schema, cast=cast, flex=flex,
                         validate=validate, **metadata)

    @property
    def id_range(self):
        return self.metadata['id_range']

    @property
    def dt_range(self):
        return self.metadata['dt_range']

    @property
    def certification(self):
        return self.metadata['certification']

    @property
    def consumption(self):
        return self.metadata['consumption']

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        if self.empty:
            return self._conform(self._data)
        df = self._data.reset_index()
        if 'id' in df:
            df['id'] = df['id'].astype('object')
        return df

    @property
    def payload(self):
        """An unindexed dataframe of payload columns."""
        if self.empty:
            return pd.DataFrame(columns=self.schema.payload)
        return self._data.reset_index(drop=True)

    def _conform(self, data, flex=True, force=False):
        """Primitive for cast, returning a non-indexed dataframe."""
        df = pd.DataFrame(data).reset_index()
        if not df.empty:
            if isinstance(self.schema, sch.MultiLogsSchema):
                df.drop_duplicates(subset=['datetime', 'label'], inplace=True)
            else:
                df.drop_duplicates(subset=['id', 'datetime'], inplace=True)
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
            missing_idxcols = [c for c in missing_columns
                               if c in self.schema.index]
            if missing_idxcols:
                msg = f"Data is missing index columns {missing_idxcols}."
                raise ConformityError(msg)
            if flex:
                schema = type(self.schema)(*self.schema.schema_columns,
                                           exclude=missing_columns)
                self.schema = schema
            else:
                msg = f"Data is missing schema columns {missing_columns}."
                raise ConformityError(msg)
        if mistyped_columns:
            dtypes = [c.dtype for c in
                      [self.columns[n] for n in mistyped_columns]]
            msg = f"Could not cast {mistyped_columns} to required " + \
                  f"dtypes {dtypes}"
            raise ConformityError(msg)
        return df[list(self.columns)]

    def cast(self, data, flex=True, force=False):
        """Casts supplied dataframe-like object to the object's schema.

        data must be a valid input to pandas.DataFrame.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a DataFrame
        * if flex is False, raises ConformityError if the data misses schema
        columns; extra columns are simply removed. If flex is True and
        columns are missing, a new schema instance will be created and
        replace self.schema -which may raise an error if mandatory columns
        are missing;
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary;
        * Verifies that all records have an id that belongs to the id_range;
        * Verifies that index columns have non-NA values
        * Verifies that the certification is anterior to the end of the
        date range.

        The force flag will silently handle incorrect inputs, such as
        unreadable datetime, and treat them as missing values.
        """
        if data is None:
            data = self.empty_frame(self.schema).reset_index()
        df = self._conform(data, flex=flex, force=force)
        if 'id' in df:
            df['id'] = pd.Categorical(df['id'], self.id_range.levels)
            if any(df['id'].isna()):
                unknowns = df['id'][df['id'].isna()].unique()
                msg = f"Unknown ids {unknowns} supplied to {self}."
                raise ConformityError(msg)
        if 'datetime' in df:
            if any(df['datetime'].isna()):
                msg = f"Missing datetime index values in {self}."
                raise ConformityError(msg)
        if 'label' in df and self.schema['label'].index:
            if any(df['label'].isna()):
                msg = f"Missing label index values in {self}."
                raise ConformityError(msg)
        if not pd.isna(self.certification):
            if not self.dt_range.upper >= self.certification:
                msg = "A certification line cannot be posterior to the " + \
                      "upper range of a data log."
                raise ConformityError(msg)
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

    def _metaslice(self, id_slice=None, dt_slice=None, xrecord=False):
        """Primitive for __getitem__, providing metadata."""
        if id_slice is None:
            id_range = self.id_range
        else:
            id_range = rge.cat_range(id_slice, sliced=self.id_range)
            id_range &= self.id_range
        if dt_slice is None:
            dt_range = self.dt_range
        elif xrecord:
            dt_range = rge.time_range(dt_slice)
        else:
            dt_range = rge.time_range(dt_slice) & self.dt_range
        metadata = self.metadata
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
        if isinstance(data, pd.DataFrame):
            data.reset_index(inplace=True)
            archetype_ = archetype(id_range, dt_range, len(data))
            if archetype_ is Record:
                try:
                    data = pd.Series(data.iloc[0])
                except IndexError:
                    raise KeyError
                data['id'] = id_range.level
                data['datetime'] = dt_range.position
        elif isinstance(data, pd.Series):
            archetype_ = archetype(id_range, dt_range)
            data['id'] = id_range.level
            data['datetime'] = dt_range.position
        else:  # Single index of single column
            archetype_ = archetype(id_range, dt_range)
            col_name = list(self.columns)[-1]
            data = {'id': id_range.level,
                    'datetime': dt_range.position,
                    col_name: data}
        if archetype_ is Record:
            metadata.pop('id_range', None)
            metadata.pop('dt_range', None)
            certification = metadata.pop('certification', pd.NaT)
            consumption = metadata.pop('consumption', pd.NaT)
            if not pd.isna(certification):
                if certification >= dt_range.position:
                    metadata['certified'] = True
            if not pd.isna(consumption):
                if consumption >= dt_range.position:
                    metadata['consumed'] = True
        else:
            if isinstance(dt_range, rge.Interval):
                upper = dt_range.upper
            else:
                upper = dt_range.position
            certification = metadata.pop('certification', pd.NaT)
            consumption = metadata.pop('consumption', pd.NaT)
            if certification:
                metadata['certification'] = min(certification, upper)
            if consumption:
                metadata['consumption'] = min(consumption, upper)
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
        return {'index': list(self.index),
                'data': self.payload.to_dict('list'),
                'schema': self.schema.to_dict(),
                'metadata': self.metadata}

    @classmethod
    def from_dict(cls, dict_, **kwargs):
        try:
            index_ = dict_['index']
            data_ = dict_['data']
            schema_ = dict_['schema']
            metadata = dict_['metadata']
        except KeyError:
            msg = f"Invalid mapping."
            raise ValueError(msg)
        if isinstance(schema_, sch.Schema):
            schema = schema_
        else:
            schema = sch.Schema.from_dict(schema_)
        data = pd.DataFrame.from_dict(data_)
        if isinstance(schema, sch.MultiLogsSchema):
            if index_:
                datetime, label = zip(*index_)
            else:
                datetime, label = [], []
            data['datetime'] = datetime
            data['label'] = label
        else:
            if index_:
                id_, datetime = zip(*index_)
            else:
                id_, datetime = [], []
            data['id'] = id_
            data['datetime'] = datetime
        id_range = rge.cat_range(metadata.pop('id_range', None))
        dt_range = rge.time_range(metadata.pop('dt_range', None))
        cls = archetype(id_range, dt_range, len(data))
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

    def certify(self, datetime):
        """Adds / update certification line."""
        self._metadata['certification'] = pd.to_datetime(datetime, utc=True)

    def __repr__(self):
        return f"<{type(self).__name__} id_range:{str(self.id_range)} " + \
               f"dt_range:{str(self.dt_range)}>"


class LogType(DataObjectType):
    _registry = dict()


class SequenceType(DataObjectType):
    _registry = dict()


class ArrayType(DataObjectType):
    _registry = dict()


class RecordSetType(DataObjectType):
    _registry = dict()


class DataLog(DataLogsBase, metaclass=LogType):
    """A doubly-index log -multiple ids and datetimes."""
    __schema__ = sch.LogsSchema
    __schema_exclusions__ = [sch.MultiLogsSchema]
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

    @classmethod
    def empty_frame(cls, schema=None):
        """Returns an empty but conform dataframe."""
        if schema is None:
            schema = cls.__schema__()
        index = pd.MultiIndex([[], []], [[], []], names=['id', 'datetime'])
        return pd.DataFrame(columns=schema.payload, index=index)

    @property
    def _empty(self):
        """Returns an empty but conform dataframe."""
        return self.empty_frame(self.schema)

    def _xdataslice(self, key, id_range, dt_range):
        if not id_range or not dt_range:
            return self._empty
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
                if dataframes:
                    return pd.concat(dataframes)
                else:
                    return self._empty
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
        xrecord = False  # flag for metadata slicing
        try:
            if isinstance(key, int):
                key = self.index[key]
                xrecord = True
            if isinstance(key, tuple):
                id_slice, dt_slice = key
            elif isinstance(key, slice):
                if any(isinstance(a, int) for a in [key.start,
                                                    key.stop,
                                                    key.step]):
                    msg = "DataLog instances don't support integer slicing."
                    raise TypeError(msg)
                else:
                    id_slice, dt_slice = None, key
                    key = (slice(None), dt_slice)
            else:
                id_slice, dt_slice = key, None
            metadata = self._metaslice(id_slice, dt_slice, xrecord=xrecord)
            id_range = metadata['id_range']
            dt_range = metadata['dt_range']
            data = self._dataslice(key, id_range, dt_range)
            return self._slice(data, metadata)
        except (ValueError, KeyError):
            raise KeyError(str(key))

    def _previous(self, id, dt):
        """Returns immediately previous datetime in index for id, or None."""
        mindex = self._data.index
        dt_index = mindex[mindex.get_loc(id)].get_level_values(1)
        if dt in dt_index:
            return dt
        dt_ix = dt_index.searchsorted(dt)
        if dt_ix > 0:
            prev_dt = dt_index[dt_ix - 1]
            if isinstance(self.schema, sch.SessionLogsSchema):
                stop = prev_dt + self._data.loc[id, prev_dt].duration
                if stop > dt:
                    return prev_dt
                else:
                    return dt
            return prev_dt
        else:
            return None


class DataSequence(DataLogsBase, metaclass=SequenceType):
    """A single-id dataframe container, indexed by datetime."""
    __schema__ = sch.LogsSchema
    __id_range__ = rge.Level
    __dt_range__ = rge.TimeInterval

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        pass

    def plot(self, staff=None, tz=None, **kwargs):
        if staff is None:
            score = plot.SimpleScore(tz=tz)
            staff = score.staves[0]
            rval = score
        else:
            rval = None
        self._plot(staff, **kwargs)
        if rval is None and tz is not None:
            staff.score.tz = tz
        return rval


class BasicDataSequence(DataSequence):
    """A single-id dataframe container, indexed by datetime."""
    __schema_exclusions__ = [sch.MultiLogsSchema]
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

    @classmethod
    def empty_frame(cls, schema=None):
        """Returns an empty but conform dataframe."""
        if schema is None:
            schema = cls.__schema__()
        index = pd.DatetimeIndex([], name='datetime')
        return pd.DataFrame(columns=schema.payload, index=index)

    @property
    def _empty(self):
        """Returns an empty but conform dataframe."""
        return self.empty_frame(self.schema)

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
        if dt_range is None:
            raise KeyError
        if self.schema.xindex:
            return self._xdataslice(key, dt_range)
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            key = slice(pd.to_datetime(start, utc=True),
                        pd.to_datetime(stop, utc=True))
        return self.data.loc[key]

    def __getitem__(self, key):
        xrecord = False  # flag for metadata slicing
        try:
            if isinstance(key, int):
                key = self.index[key]
                xrecord = True
            elif isinstance(key, slice):
                if any(isinstance(a, int) for a in [key.start,
                                                    key.stop,
                                                    key.step]):
                    ix = self.index[key]
                    if ix.empty:
                        key = slice(nxtime.MIN, nxtime.MIN)
                    else:
                        key = slice(ix[0], ix[-1])
            dt_slice = key
            metadata = self._metaslice(dt_slice=dt_slice, xrecord=xrecord)
            dt_range = metadata['dt_range']
            data = self._dataslice(key, dt_range)
            return self._slice(data, metadata)
        except (ValueError, KeyError):
            raise KeyError(str(key))

    def _previous(self, dt):
        """Returns the immediately previous datetime in the index, or None."""
        index = self._data.index
        if dt in index:
            return dt
        dt_ix = index.searchsorted(dt)
        if dt_ix > 0:
            prev_dt = index[dt_ix - 1]
            if isinstance(self.schema, sch.SessionLogsSchema):
                stop = prev_dt + self._data.loc[prev_dt].duration
                if stop > dt:
                    return prev_dt
                else:
                    return dt
            return prev_dt
        else:
            return None


class SuffixedDataSequence(DataSequence):
    """A single-id dataframe container, indexed by datetime and a label."""
    __schema__ = sch.MultiLogsSchema
    _index_columns = ['datetime', 'label']

    @property
    def id(self):
        return self.id_range.level

    @property
    def datetime(self):
        return self._data.index.get_level_values(0).copy()

    @property
    def label(self):
        return self._data.index.get_level_values(1).copy()

    @property
    def columns(self):
        cols = OrderedDict(self.schema)
        del cols['id']
        return cols

    @property
    def idx(self):
        idx_ = list(zip([self.id_range.level], self.datetime, self.label))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        df = self._data.reset_index()
        df.insert(0, 'id', self.id_range.level)
        return df

    @classmethod
    def empty_frame(cls, schema=None):
        """Returns an empty but conform dataframe."""
        if schema is None:
            schema = cls.__schema__()
        index = pd.MultiIndex([[], []], [[], []], names=['datetime', 'label'])
        return pd.DataFrame(columns=schema.payload, index=index)

    @property
    def _empty(self):
        """Returns an empty but conform dataframe."""
        return self.empty_frame(self.schema)

    def _twin_dataslice(self, key, dt_range):
        data = self.data
        data['start'] = self.datetime
        data['stop'] = data.start + data.duration
        lower, upper = dt_range.bounds
        return data[(data.start <= upper) & (data.stop > lower)]

    def _dataslice(self, key, dt_range):
        if dt_range is None:
            raise KeyError
        if self.schema.twin_index:
            return self._twin_dataslice(key, dt_range)
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            key = slice(pd.to_datetime(start, utc=True),
                        pd.to_datetime(stop, utc=True))
        return self.data.loc[key]

    def __getitem__(self, key):
        xrecord = False  # flag for metadata slicing
        try:
            if isinstance(key, int):
                key = self.index[key]
                dt_slice = key[0]
                xrecord = True
            elif isinstance(key, slice):
                if any(isinstance(a, int) for a in [key.start,
                                                    key.stop,
                                                    key.step]):
                    ix = self.index[key]
                    if ix.empty:
                        key = slice(nxtime.MIN, nxtime.MIN)
                        dt_slice = key
                    else:
                        key = slice(ix[0], ix[-1])
                        dt_lower = ix[0][0]
                        dt_upper = ix[-1][0]
                        if dt_lower == dt_upper:
                            dt_slice = dt_lower
                        else:
                            dt_slice = slice(dt_lower, dt_upper)
                else:
                    dt_slice = key
            else:
                dt_slice = key
            metadata = self._metaslice(dt_slice=dt_slice, xrecord=xrecord)
            dt_range = metadata['dt_range']
            data = self._dataslice(key, dt_range)
            if isinstance(data, pd.Series):
                data['label'] = data.name[1]
            return self._slice(data, metadata)
        except (ValueError, KeyError):
            raise KeyError(str(key))


class DataArray(DataLogsBase, metaclass=ArrayType):
    """A single datetime dataframe container, indexed by id."""
    __schema__ = sch.LogsSchema
    __schema_exclusions__ = [sch.MultiLogsSchema]
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

    @classmethod
    def empty_frame(cls, schema=None):
        """Returns an empty but conform dataframe."""
        if schema is None:
            schema = cls.__schema__()
        index = pd.Index([], name='id')
        return pd.DataFrame(columns=schema.payload, index=index)

    @property
    def _empty(self):
        """Returns an empty but conform dataframe."""
        return self.empty_frame(self.schema)

    def __getitem__(self, key):
        try:
            if isinstance(key, int):
                key = self.index[key]
            elif isinstance(key, slice):
                if any(isinstance(a, int) for a in [key.start,
                                                    key.stop,
                                                    key.step]):
                    ix = self.index[key]
                    if ix.empty:
                        key = slice('', '')
                    else:
                        key = slice(ix[0], ix[-1])
            id_slice = key
            metadata = self._metaslice(id_slice=id_slice)
            data = self.data.loc[key]
            return self._slice(data, metadata)
        except (ValueError, KeyError):
            raise KeyError(str(key))


class DataRecordSet(DataLogsBase, metaclass=RecordSetType):
    """A single-id, single-datetime dataframe container, indexed by label."""
    __schema__ = sch.MultiLogsSchema
    __id_range__ = rge.Level
    __dt_range__ = rge.TimeSingleton
    _index_columns = ['label']

    @property
    def id(self):
        return self.id_range.level

    @property
    def datetime(self):
        return self.dt_range.position

    @property
    def label(self):
        return self._data.index.copy()

    @property
    def columns(self):
        cols = OrderedDict(self.schema)
        del cols['id']
        del cols['datetime']
        return cols

    @property
    def idx(self):
        idx_ = list(zip([self.id_range.level],
                        [self.dt_range.position],
                        self.label))
        return pd.DataFrame({'idx': idx_}, index=self.index).idx

    @property
    def tabulated(self):
        """Returns a normalized, unindexed dataframe."""
        df = self._data.reset_index()
        df.insert(0, 'datetime', self.dt_range.position)
        df.insert(0, 'id', self.id_range.level)
        return df

    @classmethod
    def empty_frame(cls, schema=None):
        """Returns an empty but conform dataframe."""
        if schema is None:
            schema = cls.__schema__()
        index = pd.DatetimeIndex([], name='label')
        return pd.DataFrame(columns=schema.payload, index=index)

    @property
    def _empty(self):
        """Returns an empty but conform dataframe."""
        return self.empty_frame(self.schema)

    def __getitem__(self, key):
        try:
            if isinstance(key, int):
                key = self.index[key]
            elif isinstance(key, slice):
                if any(isinstance(a, int) for a in [key.start,
                                                    key.stop,
                                                    key.step]):
                    ix = self.index[key]
                    if ix.empty:
                        key = slice('', '')
                    else:
                        key = slice(ix[0], ix[-1])
            metadata = self.metadata
            data = self.data.loc[key]
            if isinstance(data, pd.Series):
                data['label'] = data.name
            return self._slice(data, metadata)
        except (ValueError, KeyError):
            raise KeyError(str(key))


def archetype(id_range, dt_range, cardinality=None):
    """Selects an archetype based on the type of ranges supplied."""
    if isinstance(id_range, rge.Level):
        if isinstance(dt_range, rge.TimeSingleton):
            try:
                assert cardinality > 1
            except (TypeError, AssertionError):
                return Record
            else:
                return DataRecordSet
        elif isinstance(dt_range, rge.TimeInterval) or dt_range is None:
            return DataSequence
    elif isinstance(id_range, rge.Levels) or id_range is None:
        if isinstance(dt_range, rge.TimeSingleton):
            return DataArray
        elif isinstance(dt_range, rge.TimeInterval) or dt_range is None:
            return DataLog
    else:
        raise TypeError


# =============================================================================
# Practical log types
# =============================================================================


class SampleLog(DataLog):
    __schema__ = sch.SampleLogsSchema

    @property
    def features(self):
        return [k for k, v in self.columns.items() if isinstance(v, cln.Float)]


class EventLog(DataLog):
    __schema__ = sch.EventLogsSchema


class StateLog(DataLog):
    __schema__ = sch.StateLogsSchema


class CompoundStateLog(DataLog):
    __schema__ = sch.CompoundStateLogsSchema


class SessionLog(DataLog):
    __schema__ = sch.SessionLogsSchema

    def cast(self, data, flex=True, force=False):
        df = super().cast(data, flex=flex, force=force)
        if df.empty:
            return df
        copy = df.copy()
        copy['start'] = copy.index.get_level_values(1)
        copy['stop'] = copy['start'] + copy['duration']
        grouped = copy.groupby(level='id')
        try:
            for id, g in grouped:
                if g.empty:
                    continue
                assert all(g['start'].shift(-1)[:-1] >= g['stop'][:-1])
        except AssertionError:
            msg = "Sessions cannot overlap."
            raise ConformityError(msg)
        return df

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


class SampleSequence(BasicDataSequence):
    __schema__ = sch.SampleLogsSchema

    @property
    def features(self):
        return [k for k, v in self.columns.items() if isinstance(v, cln.Float)]

    def _plot(self, staff, column, **kwargs):
        """Plot primitive, type-dependent."""
        staff.plot_sample_sequence(self, column, **kwargs)

    def plot(self, *staves, columns=None, tz=None, **kwargs):
        columns = fun.get(columns, self.features)
        if not staves:
            score = plot.RowScore(len(columns), tz=tz)
            staves = score.staves
            rval = score
        else:
            rval = None
        for col, staff in zip(columns, cycle(staves)):
            self._plot(staff, col, **kwargs)
        if rval is None and tz is not None:
            for s in staves:
                s.score.tz = tz
        return rval


class EventSequence(BasicDataSequence):
    __schema__ = sch.EventLogsSchema

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        staff.plot_event_sequence(self, **kwargs)


class MultiEventSequence(SuffixedDataSequence):
    __schema__ = sch.MultiEventLogsSchema

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        staff.plot_multi_event_sequence(self, **kwargs)

    def event_sequence(self, label, schema=None):
        """Returns an EventSequence for a single selected label."""
        data = self.data[self.label == label]
        return EventSequence(data, schema=schema, **self.metadata)


class StateSequence(BasicDataSequence):
    __schema__ = sch.StateLogsSchema

    @property
    def spans(self):
        """A dataframe of state spans.

        Adds a duration column for each state, and inserts dt_range
        metadata as needed.
        """
        df = self.data
        onsets = pd.Series(df.index)
        outsets = onsets.shift(-1)
        onsets.iloc[0] = max(onsets.iloc[0], self.dt_range.lower)
        outsets.iloc[-1] = self.dt_range.upper
        df.index = onsets
        df['duration'] = pd.Series((outsets - onsets).values, df.index)
        return df

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        staff.plot_state_sequence(self, **kwargs)


class CompoundStateSequence(BasicDataSequence):
    __schema__ = sch.CompoundStateLogsSchema

    def to_multisession_sequence(self, schema=None):
        upper = self.dt_range.upper
        sessions = []
        open_sessions = {}
        for dt, v in dict(self.labels).items():
            prev_keys = set(open_sessions)
            next_keys = set(v)
            new_keys = next_keys - prev_keys
            closing = prev_keys - next_keys
            for k in new_keys:
                session = v[k].copy()
                session['label'] = k
                if 'duration' not in session:
                    session['duration'] = upper - dt
                open_sessions[k] = session
                sessions.append(session)
            for k in closing:
                session = open_sessions.pop(k)
        return MultiSessionSequence(data=sessions, schema=schema,
                                    **self.metadata)


class SessionSequence(BasicDataSequence):
    __schema__ = sch.SessionLogsSchema

    def cast(self, data, flex=True, force=False):
        df = super().cast(data, flex=flex, force=force)
        if df.empty:
            return df
        copy = df.copy()
        copy['start'] = copy.index
        copy['stop'] = copy['start'] + copy['duration']
        try:
            assert all(copy['start'].shift(-1)[:-1] >= copy['stop'][:-1])
        except AssertionError:
            msg = "Sessions cannot overlap."
            raise ConformityError(msg)
        return df

    @property
    def start(self):
        """Start times of sessions."""
        return pd.Series(self.datetime, self.index)

    @property
    def stop(self):
        """Stop times of sessions."""
        return self.start + self.duration

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        staff.plot_session_sequence(self, **kwargs)


class MultiSessionSequence(SuffixedDataSequence):
    __schema__ = sch.MultiSessionLogsSchema

    def cast(self, data, flex=True, force=False):
        df = super().cast(data, flex=flex, force=force)
        if df.empty:
            return df
        copy = df.copy()
        copy['start'] = copy.index.get_level_values(0)
        copy['stop'] = copy['start'] + copy['duration']
        grouped = copy.groupby(level='label')
        try:
            for label, g in grouped:
                if g.empty:
                    continue
                assert all(g['start'].shift(-1)[:-1] >= g['stop'][:-1])
        except AssertionError:
            msg = "Same-label sessions cannot overlap."
            raise ConformityError(msg)
        return df

    @property
    def start(self):
        """Start times of sessions."""
        return pd.Series(self.datetime, self.index)

    @property
    def stop(self):
        """Stop times of sessions."""
        return self.start + self.duration

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        staff.plot_multi_session_sequence(self, **kwargs)

    def to_compound_state_sequence(self, schema=None):
        df = self.tabulated
        sessions = df.to_dict(orient='records')
        onsets = pd.DataFrame({'data': sessions, 'label': self.label},
                              index=self.datetime)
        outsets = pd.Series(self.label, index=self.stop).to_frame('label')
        combined = pd.concat((onsets, outsets)).sort_index()
        combined.index.name = 'datetime'
        content = {}
        labels = []
        for label, data in zip(combined.label, combined.data):
            content = content.copy()
            if pd.isna(data):
                del content[label]
            else:
                content[label] = data
            labels.append(content)
        combined['labels'] = labels
        combined = combined.reset_index().drop_duplicates('datetime',
                                                          keep='last')
        return CompoundStateSequence(data=combined[['datetime', 'labels']],
                                     schema=schema, **self.metadata)


class PeriodSequence(BasicDataSequence):
    __schema__ = sch.PeriodLogsSchema

    def _plot(self, staff, **kwargs):
        """Plot primitive, type-dependent."""
        pass


class SampleArray(DataArray):
    __schema__ = sch.SampleLogsSchema

    @property
    def features(self):
        return [k for k, v in self.columns.items() if isinstance(v, cln.Float)]


class EventArray(DataArray):
    __schema__ = sch.EventLogsSchema


class StateArray(DataArray):
    __schema__ = sch.StateLogsSchema


class CompoundStateArray(DataArray):
    __schema__ = sch.CompoundStateLogsSchema


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
