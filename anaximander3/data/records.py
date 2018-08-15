#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines data records that contain rows of columnar data.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from collections import OrderedDict, Sequence

import pandas as pd

from ..utilities import xprops, nxrange as rge
from ..utilities.jsonmixin import jsonio
from .dataobject import DataObjectType, DataObject
from .exceptions import ConformityError
from . import nxschema as sch


__all__ = []

# =============================================================================
# Base type
# =============================================================================


@jsonio
class RecordBase(DataObject):
    """Base class for data records."""
    __schema__ = None  # placeholder for specialized type parameter

    @property
    def id(self):
        return self._data.id

    @property
    def datetime(self):
        return self._data.datetime

    def cast(self, data, flex=True):
        """Casts supplied dict-like object to the object's schema.

        data must be a valid input to pandas.Series.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a Series
        * if flex is False, raises ConformityError if the data misses schema
        columns; extra columns are simply removed. If flex is True and
        columns are missing, a new schema instance will be created and
        replace self.schema -which may raise an error if mandatory columns
        are missing;
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary.
        """
        fields = OrderedDict()
        missing_fields = []
        mistyped_fields = OrderedDict()
        if data is None:
            data = []
        if isinstance(data, Sequence):
            data = dict(zip(self.schema, data))
        for name, col in self.columns.items():
            try:
                val = data[name]
            except KeyError:
                if col.default is not None:
                    fields[name] = col.default
                else:
                    missing_fields.append(name)
            else:
                try:
                    fields[name] = col.rcast(val)
                except (TypeError, ValueError):
                    mistyped_fields[name] = val
        if missing_fields:
            missing_idxflds = [c for c in missing_fields
                               if c in self.schema.index]
            if missing_idxflds:
                msg = f"Data is missing index fields {missing_idxflds}."
                raise ConformityError(msg)
            if flex:
                schema = type(self.schema)(*self.schema.schema_columns,
                                           exclude=missing_fields)
                self.schema = schema
            else:
                msg = f"Data is missing schema fields {missing_fields}."
                raise ConformityError(msg)
        elif mistyped_fields:
            types = [c.rtype for c in
                     [self.columns[n] for n in mistyped_fields]]
            mistyped = list(mistyped_fields.items())
            msg = f"Could not cast key-value pairs {mistyped} to required " + \
                  f"types {types}"
            raise ConformityError(msg)
        return pd.Series(fields)

    @property
    def idx(self):
        """Returns indexing tuple."""
        if isinstance(self.schema, sch.MultiLogsSchema):
            return (self._data.id, self._data.datetime, self._data.label)
        else:
            return (self._data.id, self._data.datetime)

    @property
    def key(self):
        """Returns row key."""
        return self.schema.rowkey(self.idx)

    @xprops.cachedproperty
    def errors(self):
        return self._validate()

    def _validate(self):
        """Runs column validators, returns a list of found errors."""
        data = self._data
        errors = []
        for name, col in self.columns.items():
            for i, v in enumerate(col.validators):
                valid = v(self, col, data[name])
                if valid is not True:
                    errors.append(valid)
        return errors

    def validate(self):
        """Returns True or False whether the data validates or not."""
        return not self._validate()

    @property
    def tabulated(self):
        """Returns a normalized, ordered dictionary."""
        return OrderedDict((c, getattr(self, c)) for c in self.columns)

    @property
    def payload(self):
        """Returns a normalized, ordered dictionary with paylod columns."""
        return OrderedDict((c, getattr(self, c)) for c in self.schema.payload)

    def to_dict(self, **kwargs):
        return {'index': self.idx,
                'data': self.payload,
                'schema': self.schema.to_dict(),
                'metadata': self.metadata}

    def to_data_dict(self):
        return {'index': self.idx,
                'data': self.payload}

    @classmethod
    def from_dict(cls, dict_, **kwargs):
        try:
            index_ = dict_['index']
            data_ = dict_['data']
            schema_ = dict_['schema']
            metadata = dict_.get('metadata', {})
        except KeyError:
            msg = f"Invalid mapping."
            raise ValueError(msg)
        if isinstance(schema_, sch.Schema):
            schema = schema_
        else:
            schema = sch.Schema.from_dict(schema_)
        validate = kwargs.get('validate', False)
        if isinstance(schema, sch.MultiLogsSchema):
            data_['id'], data_['datetime'], data_['label'] = index_
        else:
            data_['id'], data_['datetime'] = index_
        return cls(data_, schema=schema, validate=validate, **metadata)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self._data.equals(other._data) and \
            self.metadata == other.metadata

    def copy(self, id=None):
        """Copies record, optionally with a different id -used for testing."""
        d = self.to_dict()
        if id is not None:
            d['index'] = (id, d['index'][1])
        return self.from_dict(d)


class RecordType(DataObjectType):
    _registry = dict()


class Record(RecordBase, metaclass=RecordType):
    __schema__ = sch.LogsSchema

    def __repr__(self):
        return f"<{type(self).__name__} id:{self.id} " + \
               f"datetime:{str(self.datetime)}>"

# =============================================================================
# Practical record types
# =============================================================================


class SampleRecord(Record):
    __schema__ = sch.SampleLogsSchema


class EventRecord(Record):
    __schema__ = sch.EventLogsSchema

    def __repr__(self):
        return f"<{type(self).__name__} id:{self.id} " + \
               f"datetime:{str(self.datetime)} label:{self.label}>"


class StateRecord(Record):
    __schema__ = sch.StateLogsSchema

    def nulled_copy(self, datetime=None):
        """Return a record with the label set to None.

        if datetime is None, the copy has the same datetime as the original,
        otherwise it is modified accordingly.
        """
        dict_ = self.to_dict()
        dict_['data']['label'] = None
        if datetime is not None:
            dt = pd.to_datetime(datetime, utc=True)
            dict_['index'] = (self.id, dt)
        return self.from_dict(dict_)

    def __repr__(self):
        return f"<{type(self).__name__} id:{self.id} " + \
               f"datetime:{str(self.datetime)} label:{self.label}>"


class SessionRecord(Record):
    __schema__ = sch.SessionLogsSchema

    @property
    def start(self):
        """Start times of sessions."""
        return self.datetime

    @property
    def stop(self):
        """Stop times of sessions."""
        return self.datetime + self.duration

    @property
    def span(self):
        return rge.TimeInterval(self.start, self.stop)

    def __repr__(self):
        return f"<{type(self).__name__} id:{self.id} " + \
               f"datetime:{str(self.datetime)} label:{self.label}>"


class PeriodRecord(Record):
    __schema__ = sch.PeriodLogsSchema
