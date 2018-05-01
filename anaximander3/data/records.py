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

from collections import OrderedDict

import pandas as pd

from ..utilities import xprops
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

    def cast(self, data):
        """Casts supplied dict-like object to the object's schema.

        data must be a valid input to pandas.Series.
        The method performs the following functions:
        * raises PandasError if data cannot be cast to a Series
        * raises ConformityError if the data misses schema columns;
        extra columns are simply removed.
        * recasts columns to the dtype specified in the schema if necessary;
        * reorders columns to match the schema if necessary.
        """
        fields = OrderedDict()
        missing_fields = []
        mistyped_fields = OrderedDict()
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
        """Returns a series of indexing tuples."""
        return (self._data.id, self._data.datetime)

    @property
    def key(self):
        """Returns a series of row keys, indexed by self's index."""
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

    def to_dict(self, **kwargs):
        return {'data': self.tabulated,
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
        schema = sch.Schema.from_dict(schema_)
        validate = kwargs.get('validate', False)
        return cls(data_, schema=schema, validate=validate, **metadata)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self._data.equals(other._data) and \
            self.metadata == other.metadata


class RecordType(DataObjectType):
    _registry = dict()


class Record(RecordBase, metaclass=RecordType):
    __schema__ = sch.LogsSchema


# =============================================================================
# Practical record types
# =============================================================================


class SampleRecord(Record):
    __schema__ = sch.SampleLogsSchema


class EventRecord(Record):
    __schema__ = sch.EventLogsSchema


class StateRecord(Record):
    __schema__ = sch.StateLogsSchema


class SessionRecord(Record):
    __schema__ = sch.SessionLogsSchema


class PeriodRecord(Record):
    __schema__ = sch.PeriodLogsSchema
