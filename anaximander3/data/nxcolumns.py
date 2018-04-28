#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines basic column types used in data structures.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from collections import Mapping, Iterable
from itertools import count

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype

from ..utilities import nxtime
from ..meta.nxdescriptors import Registrable


__all__ = []

# =============================================================================
# Column base types
# =============================================================================


class NxColumn(Registrable):
    """NxColumns are schema descriptors.

    NxColumn inherit name, cls and registration_id attributes from Registrable.
    NxColumn can be marked as index: index columns are featured in a schema's
    index whereas other are payload columns (or column families in the case
    of a multi-schema). Index columns come in two flavors: sequential or
    nominal. A sequential index sets ordering between records that otherwise
    share the same nominal index values.
    A payload column can be marked as required, in which case instances of a
    Schema type that features that column must include the column. Columns
    that are not marked as required may be excluded -the primary use case
    being queries that focus on particular columns. Evidently, index
    columns are always required -hence the required flag is ignored by
    index columns.
    Default is used to fill in values that are not supplied in the
    initialization of a record.
    Missing is used to fill in missing data in a dataframe. If default is
    specified, then it is unnecessary to specify missing -the default value
    will be used. However there are cases where it might be OK for records
    to store no value for a given field. In that case, specify default to
    None and provide a missing value to ensure consistency in dataframes.
    If no missing value is provided, pandas will assing NaN, but that might
    interfere with the ability to cast the dataframe to the desired dtypes.
    Validate is an optional callable or list of callables that are
    applied sequentially. Each callable must have the following signature:
        * The record being validated, in which fields can be accessed
        either as instance variables or map keys;
        * The NxColumn instance itself;
        * The value that must be validated.
        * optionally it can return a formatted string used to pubish error
        messages when validation fails.
    """
    __registry__ = '__nxcolumns__'
    __counter__ = count()
    dtype = None
    rtype = None
    strict_ordering = False

    def __init__(self, *, index=None, required=False, default=None,
                 missing=None, validate=None):
        super().__init__()
        self.index = index
        if index:
            self.required = True
            default = missing = None
        else:
            self.required = required
        self.default = default
        if default is not None:
            self.missing = self.default
        else:
            self.missing = missing
        if validate is not None:
            if isinstance(validate, Iterable):
                self.validators = list(validate)
            else:
                self.validators = [validate]
        else:
            self.validators = []

    def match_type(self, typemap):
        """Returns the type corresponding to self in a type map."""
        for t in type(self).__mro__:
            try:
                match = typemap[t]
            except KeyError:
                continue
            else:
                if isinstance(match, TypeMapper):
                    return match(self)
                else:
                    return match
        raise TypeError(f"Could not match {self} in {typemap}.")

    def validator(self, method):
        """Wraps a schema method to add a validator."""
        self.validators.append(method)

    def rcast(self, value, force=False):
        """Casts a scalar to the appropriate type."""
        try:
            return self.rtype(value)
        except (TypeError, ValueError):
            if force:
                return self.missing
            else:
                raise

    def dcast(self, series, force=False):
        """Casts a series to the appropriate type."""
        try:
            return series.astype(self.dtype)
        except (TypeError, ValueError):
            if force:
                return pd.Series([self.missing] * len(series),
                                 index=series.index)
            else:
                raise


class Numeric(NxColumn):
    """Base class for numeric types."""
    dtype = np.dtype('float64')
    rtype = np.float64

    def __init__(self, *, index=None, required=False, default=None,
                 missing=None, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)


class Integer(Numeric):
    """Integer column type.

    NOTE: this column type does not admit missing values, hence it is
    mandatory that an integer default be provided. If that conflicts with
    the use case for the column, select a different type.
    """
    dtype = np.dtype('int64')
    rtype = np.int64

    def __init__(self, *, index=None, required=False, default=0,
                 missing=None, validate=None):
        if not isinstance(default, int):
            msg = "Integer column type can only accepts integers as default."
            raise TypeError(msg)
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)


class Bool(NxColumn):
    """Boolean column type.

    NOTE: this column type does not admit missing values, hence it is
    mandatory that a boolean default be provided. If that conflicts with
    the use case for the column, select a different type.
    """
    dtype = np.dtype('bool')
    rtype = np.bool_

    def __init__(self, *, index=None, required=False, default=False,
                 missing=None, validate=None):
        if not isinstance(default, bool):
            msg = "Boolean column type can only accepts booleans as default."
            raise TypeError(msg)
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)

    def rcast(self, value, force=False):
        """Casts a scalar to the appropriate type."""
        if isinstance(value, (bool, np.bool_)):
            return np.bool_(value)
        elif value in ('True', 'true', 'TRUE', 'T'):
            return True
        elif value in ('False', 'false', 'FALSE', 'F'):
            return False
        elif force:
            return False
        else:
            raise ValueError

    def dcast(self, series, force=False):
        """Casts a series to the appropriate type."""
        series.replace(['True', 'true', 'TRUE', 'T'], True, inplace=True)
        series.replace(['False', 'false', 'FALSE', 'F'], False, inplace=True)
        try:
            return series.astype('bool')
        except (TypeError, ValueError):
            if force:
                return pd.Series([False] * len(series), index=series.index)
            else:
                raise


class Float(Numeric):
    """Float column type."""
    pass


class String(NxColumn):
    """Base class for text-based field columns."""
    dtype = np.dtype('object')
    rtype = str

    def __init__(self, *, index=None, required=False, default=None,
                 missing=None, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)


class Text(String):
    """Text column type."""

    def __init__(self, *, index=None, required=False, default='',
                 missing=None, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)


class Categorical(NxColumn):
    """Categorical column type, with string-defined categories.

    The ordered flag specifies whether the categories define a natural sort
    order.
    """
    dtype = 'category'
    rtype = str

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=False, default=None, missing=None, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)
        if categories is not None:
            self.categories = tuple(categories)
            self.dtype = CategoricalDtype(categories, ordered)
        else:
            self.categories = None

    def rcast(self, value, force=False):
        """Casts a scalar to the appropriate type."""
        if value in (None, np.nan):
            return None
        if self.categories is not None:
            if value not in self.categories:
                raise ValueError
        return str(value)


class StateLabel(Categorical):
    """State label column type."""

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=True, default=None, missing=None, validate=None):
        super().__init__(categories, ordered, index=index, required=required,
                         default=default, missing=missing, validate=validate)


class EventLabel(Categorical):
    """Column specifying event label"""

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=True, default=None, missing=None, validate=None):
        super().__init__(categories, ordered, index=index, required=required,
                         default=default, missing=missing, validate=validate)


class SessionLabel(Categorical):
    """Column specifying session label"""

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=True, default=None, missing=None, validate=None):
        super().__init__(categories, ordered, index=index, required=required,
                         default=default, missing=missing, validate=validate)


class DateTimeBase(NxColumn):
    """Base class for tz-aware date & time columns."""
    dtype = np.dtype('datetime64[ns]')
    rtype = pd.Timestamp

    def __init__(self, tz='UTC', *, index=False, required=False,
                 default=None, missing=pd.NaT, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)
        self.tz = tz
        self.dtype = DatetimeTZDtype(tz=tz, unit='ns')

    def rcast(self, value, force=False):
        """Casts a scalar to the appropriate type."""
        timestamp = pd.to_datetime(value, utc=True,
                                   errors='coerce' if force else 'raise')
        return timestamp if self.tz == 'UTC' else timestamp.tz_convert(self.tz)

    def dcast(self, series, force=False):
        """Casts a series to the appropriate type."""
        series = pd.to_datetime(series, utc=True,
                                errors='coerce' if force else 'raise')
        return series if self.tz == 'UTC' else series.dt.tz_convert(self.tz)


class Date(DateTimeBase):
    """Date column type."""
    pass


class Time(DateTimeBase):
    """Time column type."""
    pass


class DateTime(DateTimeBase):
    """DateTime column type."""
    pass


class Timestamp(Integer):
    """Timestamp column type."""
    pass


class TimeDelta(NxColumn):
    """Time delta column type."""
    dtype = np.dtype('timedelta64[ns]')
    rtype = pd.Timedelta

    def __init__(self, *, index=False, required=False,
                 default=None, missing=pd.NaT, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)

    def rcast(self, value, force=False):
        """Casts a scalar to the appropriate type."""
        return pd.to_timedelta(value, errors='coerce' if force else 'raise')

    def dcast(self, series, force=False):
        """Casts a series to the appropriate type."""
        return pd.to_timedelta(series, errors='coerce' if force else 'raise')


class ObjectID(Integer):
    """Column type for integer references to objects.

    Requires an object type at instantiation.
    """

    def __init__(self, otype, *, index=None, required=False, default=0,
                 missing=None, validate=None):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate)
        self.otype = otype

# =============================================================================
# Type map class
# =============================================================================


class TypeMapper:
    """A callable designed to map a column to a parametric type.

    The supplied function should take a single column argument.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, column):
        return self.func(column)


class TypeMap(Mapping):
    """A mapping of column types to another type system."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return self._mapping.__iter__()

    def __len__(self):
        return self._mapping.__len__()

    def __call__(self, column):
        """Matches a given column to a mapped type."""
        return column.match_type(self)
