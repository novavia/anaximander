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
from collections.abc import Set, Iterable, Sequence
from itertools import chain
from numbers import Number

from google.cloud.bigtable.row_filters import ValueRangeFilter, \
    ColumnQualifierRegexFilter, RowFilterChain
import numpy as np
import pandas as pd

from .nxtime import datetime
from . import nxattr, xprops
from .functions import get, passthrough, pairwise


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
    pass


class MultiRange(Range):
    """Abastract base class for Range unions."""
    pass


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

    # XXX: A more robust implementation would involve creating an
    # empty interval object.
    @classmethod
    def intersection(cls, a, b):
        """Returns an interval or None."""
        lower = max([a.lower, b.lower])
        upper = min([a.upper, b.upper])
        if upper < lower:
            return None
        return cls(lower, upper)

    def __and__(self, other):
        """Implements intersection at the instance level."""
        return self.intersection(self, other)


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

    def tz_convert(self, tzinfo):
        return type(self)(self.lower.tz_convert(tzinfo),
                          self.upper.tz_convert(tzinfo))

    @property
    def duration(self):
        return self.upper - self.lower


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


# =============================================================================
# Multi-Interval object
# =============================================================================

# Utility classes for normalizing Multi-Intervals


class Bound:
    """An interval's bound."""

    def __init__(self, interval, position):
        self.interval = interval
        self.position = position


class Lower(Bound):
    """An interval's lower bound."""
    pass


class Upper(Bound):
    """An interval's upper bound."""
    pass


def bounds(interval):
    return Lower(interval, interval.lower), Upper(interval, interval.upper)


class MultiInterval(MultiRange, Sequence):
    """An object representing a union of intervals."""
    tolerance = None  # Optional tolerance for interval overlap

    def __init__(self, intervals=None):
        intervals = get(intervals, [])
        self._intervals = self.normalize(*intervals)

    @classmethod
    def normalize(cls, *intervals):
        """Normalizes intervals into an ordered, non-overlapping set."""
        intervals = list(intervals)
        if not intervals:
            return []
        itype = type(intervals[0])
        series = pd.Series((1, -1) * len(intervals),
                           index=chain(*[i.bounds for i in intervals]))
        series.sort_index(inplace=True)
        series.index.name = 'idx'
        df = series.to_frame('vals').reset_index()
        grouped = df.groupby('idx')
        series = grouped.aggregate(np.sum).vals
        signature = series.cumsum()
        is_upper = signature == 0.0
        uppers = signature.index[is_upper]
        is_lower = is_upper.shift().fillna(True)
        lowers = signature.index[is_lower]
        if cls.tolerance is not None:
            lowest, uppest = lowers[0], uppers[-1]
            interior = lowers[1:] - uppers[:-1]
            keepers = interior > cls.tolerance
            lowers = [lowest] + list(lowers[1:][keepers])
            uppers = list(uppers[:-1][keepers]) + [uppest]
        return [itype(lower, upper) for lower, upper in zip(lowers, uppers)]

    @classmethod
    def intersection(cls, a, b):
        """Returns the intersection of two MultiIntervals, as a MultiInterval.

        This method assumes that a and b are MultiInterval instances,
        and, crtically, that they have already been normalized.
        """
        if not a:
            return cls()
        itype = type(a._intervals[0])
        series_a = pd.Series((1, -1) * len(a),
                             index=chain(*[i.bounds for i in a]))
        series_b = pd.Series((1, -1) * len(b),
                             index=chain(*[i.bounds for i in b]))
        dfa = series_a.to_frame('a')
        dfb = series_b.to_frame('b')
        df = dfa.merge(dfb, how='outer', left_index=True, right_index=True)
        merged = df.fillna(0)
        signature = merged.a.cumsum() + merged.b.cumsum()
        is_lower = signature == 2.0
        lowers = signature.index[is_lower]
        uppers = signature.index[is_lower.shift().fillna(False)]
        return cls([itype(l, u) for l, u in zip(lowers, uppers)])

    @classmethod
    def difference(cls, a, b):
        """Returns the difference of two MultiIntervals, as a MultiInterval.

        This method assumes that a and b are MultiInterval instances,
        and, crtically, that they have already been normalized.
        """
        if not a:
            return cls()
        itype = type(a._intervals[0])
        series_a = pd.Series((1, -1) * len(a),
                             index=chain(*[i.bounds for i in a]))
        series_b = pd.Series((1, -1) * len(b),
                             index=chain(*[i.bounds for i in b]))
        dfa = series_a.to_frame('a')
        dfb = series_b.to_frame('b')
        df = dfa.merge(dfb, how='outer', left_index=True, right_index=True)
        merged = df.fillna(0)
        signature = (merged.a.cumsum() - merged.b.cumsum()). \
            apply(lambda x: max((x, 0.0)))
        is_lower = signature == 1.0
        lowers = signature.index[is_lower]
        uppers = signature.index[is_lower.shift().fillna(False)]
        return cls([itype(l, u) for l, u in zip(lowers, uppers)])

    @classmethod
    def union(cls, *instances):
        """Returns the union of MultiIntervals, as a MultiInterval."""
        intervals = chain(*(i._intervals for i in instances))
        return cls(intervals)

    def __getitem__(self, ix):
        return self._intervals.__getitem__(ix)

    def __len__(self):
        return self._intervals.__len__()

    def __eq__(self, other):
        return self._intervals == list(other)

    def __and__(self, other):
        return self.intersection(self, other)

    def __sub__(self, other):
        return self.difference(self, other)

    def __or__(self, other):
        return self.union(self, other)

    def __contains__(self, item):
        return any(item in i for i in self._intervals)

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, self._intervals)


class MultiTimeInterval(MultiInterval):
    tolerance = pd.Timedelta(microseconds=1)

    @property
    def duration(self):
        if not self:
            return pd.Timedelta(0)
        durations = pd.Series(i.length for i in self)
        return durations.sum()

    @property
    def gapspace(self):
        gaps = type(self)([TimeInterval(a.upper, b.lower)
                           for a, b in pairwise(self)])
        return gaps.duration

    @property
    def compact(self):
        if not self:
            return None
        return TimeInterval(self[0].lower, self[-1].upper)
