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
from copy import deepcopy
from itertools import count
import re

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.core.dtypes.dtypes import DatetimeTZDtype

from ..utilities.jsonmixin import jsonio
from ..meta.nxdescriptors import Registrable


__all__ = []

# =============================================================================
# Validator types
# =============================================================================


class Validator:
    """Validators are callables used to validate records.

    To use a validator, pass an instance or list of instances to the
    validate argument of NxColumn's constructor.
    To define a new validator, write the __call__ function, which should
    either return True or a diagnostic string.
    if nullable is True, then the missing value of the column will pass
    validation, else it fails.
    Additional wildcards can be passed as an iterable, or by composing the
    validator with the wildcard method. Wildcards always pass, whereas
    the missing value only passes if nullable is True.
    """

    def __init__(self, nullable=True, wildcards=None):
        self.nullable = nullable
        if wildcards is not None:
            self.wildcards = list(wildcards)
        else:
            self.wildcards = []

    def __call__(self, record, column, value):
        if self.nullable:
            if value is column.missing or value == column.missing:
                return True
        if value in self.wildcards:
            return True

    def copy(self):
        """Returns a copy of self -may need overwriting depending on init."""
        copy_ = type(self)
        copy_.__dict__ = deepcopy(self.__dict__)
        return copy_

    def wildcard(self, value):
        """Returns a new instance with the wildcard added."""
        copy_ = self.copy()
        copy_.wildcards.append(value)
        return copy_


class RangeValidator(Validator):

    def __init__(self, lower=None, upper=None, nullable=True, wildcards=None):
        self.lower = lower
        self.upper = upper
        super().__init__(nullable, wildcards)

    def _lower_bound_validator(self, record, column, value):
        if value < self.lower:
            return f"{value} is less than lower bound {self.lower}."
        else:
            return True

    def _upper_bound_validator(self, record, column, value):
        if value > self.upper:
            return f"{value} is more than upper bound {self.upper}."
        else:
            return True

    def _dual_bound_validator(self, record, column, value):
        if value < self.lower or value > self.upper:
            return f"{value} must lie between {self.lower} and {self.upper}."
        else:
            return True

    def __call__(self, record, column, value):
        if super().__call__(record, column, value) is True:
            return True
        if self.lower is None:
            if self.upper is None:
                return True
            else:
                return self._lower_bound_validator(record, column, value)
        elif self.upper is None:
            return self._upper_bound_validator(record, column, value)
        else:
            return self._dual_bound_validator(record, column, value)


class MembershipValidator(Validator):

    def __init__(self, enum, nullable=True, wildcards=None):
        self.enum = list(enum)
        super().__init__(nullable, wildcards)

    def __call__(self, record, column, value):
        if super().__call__(record, column, value) is True:
            return True
        if value in self.enum:
            return True
        else:
            return f"{value} must be a member of {self.enum}."


class RegExValidator(Validator):

    def __init__(self, pattern, nullable=False, wildcards=None):
        self.regex = re.compile(pattern)
        super().__init__(nullable, wildcards)

    def __call__(self, record, column, value):
        if super().__call__(record, column, value) is True:
            return True
        if self.regex.match(value):
            return True
        else:
            f"{value} does not match pattern {self.regex.pattern}."

# =============================================================================
# Column base types
# =============================================================================


@jsonio
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
    ctype = 'column'  # type name used in serialization
    _serial_attrs = []  # attributes to include in serialization

    def __init__(self, *, index=None, required=False, default=None,
                 missing=None, validate=None, **metadata):
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
        self.metadata = metadata

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

    def to_dict(self, **kwargs):
        """Serialization format (one-way only)."""
        dict_ = {'ctype': self.ctype}
        if self.index:
            dict_['index'] = self.index
        dict_.update({k: getattr(self, k, None) for k in self._serial_attrs})
        dict_.update(self.metadata)
        return dict_


class Numeric(NxColumn):
    """Base class for numeric types."""
    pass


class Integer(Numeric):
    """Integer column type.

    NOTE: this column type does not admit missing values, hence it is
    mandatory that an integer default be provided. If that conflicts with
    the use case for the column, select a different type.
    """
    dtype = np.dtype('int64')
    rtype = np.int64
    ctype = 'int'

    def __init__(self, *, index=None, required=False, default=0,
                 validate=None, **metadata):
        if not isinstance(default, int):
            msg = "Integer column type can only accepts integers as default."
            raise TypeError(msg)
        super().__init__(index=index, required=required, default=default,
                         validate=validate, **metadata)


class Bool(NxColumn):
    """Boolean column type.

    NOTE: this column type does not admit missing values, hence it is
    mandatory that a boolean default be provided. If that conflicts with
    the use case for the column, select a different type.
    """
    dtype = np.dtype('bool')
    rtype = np.bool_
    ctype = 'bool'

    def __init__(self, *, index=None, required=False, default=False,
                 validate=None, **metadata):
        if not isinstance(default, bool):
            msg = "Boolean column type can only accepts booleans as default."
            raise TypeError(msg)
        super().__init__(index=index, required=required, default=default,
                         validate=validate, **metadata)

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
    """Float column type. Features optional decimals argument."""
    dtype = np.dtype('float64')
    rtype = np.float64
    ctype = 'float'
    _serial_attrs = ['decimals']

    def __init__(self, decimals=None, *, index=None, required=False,
                 default=None, validate=None, **metadata):
        super().__init__(index=index, required=required, default=default,
                         missing=np.nan, validate=validate, **metadata)
        self.decimals = decimals

    def rcast(self, value, force=False):
        """Casts a scalar to the appropriate type."""
        try:
            if self.decimals is not None:
                return np.around(value, self.decimals)
            else:
                return np.float64(value)
        except (TypeError, ValueError):
            if force:
                return np.nan
            else:
                raise

    def dcast(self, series, force=False):
        """Casts a series to the appropriate type."""
        try:
            if self.decimals is not None:
                return series.astype('float').round(self.decimals)
            else:
                return series.astype('float')
        except (TypeError, ValueError):
            if force:
                return pd.Series([np.nan] * len(series), index=series.index)
            else:
                raise


class Measurement(Float):
    """Specialized float for physical measurements."""
    ctype = 'measurement'
    _serial_attrs = ['decimals', 'units']

    def __init__(self, decimals=None, units=None, *, index=None,
                 required=False, default=None, validate=None, **metadata):
        super().__init__(decimals=decimals, index=index, required=required,
                         default=default, validate=validate, **metadata)
        self.units = units


class Percentage(Float):
    """Specialized percentage type."""
    ctype = 'percentage'

    def __init__(self, decimals=1, *, index=None, required=False,
                 default=None, validate=None, **metadata):
        super().__init__(decimals, index=index, required=required,
                         default=default, validate=validate, **metadata)
        self.validator(RangeValidator(0., 100.))


class String(NxColumn):
    """Base class for text-based field columns."""
    dtype = np.dtype('object')
    rtype = str
    ctype = 'str'

    def __init__(self, *, index=None, required=False, default=None,
                 missing=np.nan, validate=None, **metadata):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate,
                         **metadata)


class Text(String):
    """Text column type."""
    ctype = 'text'

    def __init__(self, *, index=None, required=False, default='',
                 missing=None, validate=None, **metadata):
        super().__init__(index=index, required=required,
                         default=default, missing=missing, validate=validate,
                         **metadata)


class Categorical(NxColumn):
    """Categorical column type, with string-defined categories.

    The ordered flag specifies whether the categories define a natural sort
    order.
    """
    dtype = 'category'
    rtype = str
    ctype = 'category'

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=False, default=None, validate=None, **metadata):
        super().__init__(index=index, required=required,
                         default=default, missing=np.nan, validate=validate,
                         **metadata)
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
    ctype = 'state'

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=True, default=None, validate=None, **metadata):
        super().__init__(categories, ordered, index=index, required=required,
                         default=default, validate=validate, **metadata)


class EventLabel(Categorical):
    """Column specifying event label"""
    ctype = 'event'

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=True, default=None, validate=None, **metadata):
        super().__init__(categories, ordered, index=index, required=required,
                         default=default, validate=validate, **metadata)


class SessionLabel(Categorical):
    """Column specifying session label"""
    ctype = 'session'

    def __init__(self, categories=None, ordered=False, *, index=None,
                 required=True, default=None, validate=None, **metadata):
        super().__init__(categories, ordered, index=index, required=required,
                         default=default, validate=validate, **metadata)


class DateTimeBase(NxColumn):
    """Base class for tz-aware date & time columns."""
    dtype = np.dtype('datetime64[ns]')
    rtype = pd.Timestamp

    def __init__(self, tz='UTC', *, index=False, required=False,
                 default=None, validate=None, **metadata):
        super().__init__(index=index, required=required, default=default,
                         missing=pd.NaT, validate=validate, **metadata)
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
    ctype = 'date'


class Time(DateTimeBase):
    """Time column type."""
    ctype = 'time'


class DateTime(DateTimeBase):
    """DateTime column type."""
    ctype = 'datetime'


class Timestamp(Integer):
    """Timestamp column type."""
    ctype = 'timestamp'


class TimeDelta(NxColumn):
    """Time delta column type."""
    dtype = np.dtype('timedelta64[ns]')
    rtype = pd.Timedelta
    ctype = 'timedelta'

    def __init__(self, *, index=False, required=False, default=None,
                 validate=None, **metadata):
        super().__init__(index=index, required=required, default=default,
                         missing=pd.NaT, validate=validate, **metadata)

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
    ctype = 'object'

    def __init__(self, otype, *, index=None, required=False, default=0,
                 validate=None, **metadata):
        super().__init__(index=index, required=required,
                         default=default, validate=validate, **metadata)
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
