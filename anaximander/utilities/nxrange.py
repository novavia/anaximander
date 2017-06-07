#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines ranges for data selection and querying.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections.abc import Set, Iterable
from numbers import Number

from google.cloud.bigtable.row_filters import ValueRangeFilter, \
    ColumnQualifierRegexFilter, RowFilterChain

from .nxtime import datetime
from . import nxattr, xprops
from .functions import passthrough


__all__ = ['float_interval', 'time_interval', 'string_interval', 'levels']

# =============================================================================
# Class declarations
# =============================================================================


class Range(abc.ABC):
    """Abstract base class for all Range objects."""


class ContinuousRange(Range):
    """Abstract base class for Ranges in continuous data dimensions."""
    pass


class DiscreteRange(Range):
    """Abstract base class for Ranges in discrete data dimensions."""


def _sqlstring(val):
    """Makes val into a string, single-quoted or unquoted as appropriate."""
    if isinstance(val, Number):
        return str(val)
    else:
        return "'{0}'".format(val)


@nxattr.s(init=False, these={'lower': nxattr.ib(), 'upper': nxattr.ib()})
class Interval(ContinuousRange, Iterable):
    """For now intervals are closed."""
    __lower_convert__ = None
    __upper_convert__ = None

    def __init__(self, lower=None, upper=None):
        self._lower_input = lower
        self._lower = self.__lower_convert__(lower)
        self._upper_input = upper
        self._upper = self.__upper_convert__(upper)
        if self.lower > self.upper:
            msg = "Cannot set interval with lower bound greater " + \
                "than upper bound."
            raise ValueError(msg)

    @xprops.cachedproperty
    def lower(self):
        return None

    @xprops.cachedproperty
    def upper(self):
        return None

    @property
    def length(self):
        return self.upper - self.lower

    @property
    def bounds(self):
        return (self.lower, self.upper)

    def __iter__(self):
        return iter((self.lower, self.upper))

    def __contains__(self, item):
        if isinstance(item, Interval):
            return item.lower >= self.lower and item.upper <= self.upper
        else:
            return item >= self.lower and item <= self.upper

    def sql(self, attr):
        """Returns a sql statement fragment making attr within self."""
        if self._lower_input is not None:
            lower = attr + " >= " + _sqlstring(self.lower)
        else:
            lower = None
        if self._upper_input is not None:
            upper = attr + " <= " + _sqlstring(self.upper)
        else:
            upper = None
        return " AND ".join((s for s in (lower, upper) if s is not None))

    def btfilter(self, attr):
        """Returns bigtable row filter applying self's range to target cell."""
        colfilter = ColumnQualifierRegexFilter(attr.encode('utf-8'))
        if self._lower_input is None:
            lower = None
        else:
            lower = str(self.lower).encode('utf-8')
        if self._upper_input is None:
            upper = None
        else:
            upper = str(self.upper).encode('utf-8')
        rgefilter = ValueRangeFilter(lower, upper)
        return RowFilterChain([colfilter, rgefilter])


def _lower_float_convert(value):
    if value is None:
        return float('-inf')
    else:
        return float(value)


def _upper_float_convert(value):
    if value is None:
        return float('inf')
    else:
        return float(value)


class FloatInterval(Interval):
    """An interval of floats."""
    __lower_convert__ = staticmethod(_lower_float_convert)
    __upper_convert__ = staticmethod(_upper_float_convert)


def _lower_time_convert(value):
    if value is None:
        return datetime.min
    else:
        return datetime(value)


def _upper_time_convert(value):
    if value is None:
        return datetime.max
    else:
        return datetime(value)


class TimeInterval(Interval):
    """An interval of datetimes.

    The arguments are automatically converted to pandas Timestamp if
    possible, potentially raising an error if that is not possible.
    Naive datetime values are also automatically converted to UTC.
    Finally, None is an admissible value for either lower or upper, in
    which case it will be converted to anaximander's absolute time bounds,
    currently set at Jan. 1 1970, UTC and Jan. 1 2100, UTC.
    """
    __lower_convert__ = staticmethod(_lower_time_convert)
    __upper_convert__ = staticmethod(_upper_time_convert)


def _lower_string_convert(value):
    if value is None:
        return ''
    else:
        return str(value)


def _upper_string_convert(value):
    if value is None:
        # Maximum allowable argument to chr
        # Technically a string that starts with this character would be
        # greater than the purported max value created below, but that is
        # about as likely as snow in the tropics.
        return chr(1114111)
    else:
        return str(value)


class StringInterval(Interval):
    """An interval of strings."""
    __lower_convert__ = staticmethod(_lower_string_convert)
    __upper_convert__ = staticmethod(_upper_string_convert)


class Levels(DiscreteRange, Set):
    """Holds a set of discrete levels."""

    def __init__(self, levels):
        self._levels = set(levels)

    def __contains__(self, item):
        return self._levels.__contains__(item)

    def __iter__(self):
        return self._levels.__iter__()

    def __len__(self):
        return self._levels.__len__()

    def __eq__(self, other):
        return self._levels == set(other)

    def __repr__(self):
        return "Levels({0})".format(repr(self._levels))

    def sql(self, attr):
        """Returns a sql statement fragment making attr within self."""
        return attr + " IN (" + ", ".join(_sqlstring(l) for l in self) + ")"


class Level(Levels):
    """Holds a single level."""

    def __init__(self, level):
        super().__init__([level])
        self._level = level

    def __eq__(self, other):
        return self._level.__eq__(other)

    def __repr__(self):
        return "Level({0})".format(repr(self._level))

    def sql(self, attr):
        """Returns a sql statement fragment making attr equals to self."""
        return attr + " = " + _sqlstring(self._level)

# =============================================================================
# Helper functions
# =============================================================================


@passthrough(FloatInterval)
def float_interval(lower=None, upper=None):
    """Creates or passes through a float interval from lower, upper bound."""
    return FloatInterval(lower, upper)


@passthrough(TimeInterval)
def time_interval(lower=None, upper=None):
    """Creates or passes through a time interval from lower, upper bound."""
    return TimeInterval(lower, upper)


@passthrough(StringInterval)
def string_interval(lower=None, upper=None):
    """Creates or passes through a string interval from lower, upper bound."""
    return StringInterval(lower, upper)


@passthrough(Level, Levels)
def levels(arg):
    """Creates or passes through either a Level or Levels."""
    if isinstance(arg, Iterable) and not isinstance(arg, str):
        return Levels(arg)
    return Level(arg)
