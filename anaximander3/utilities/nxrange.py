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
from functools import partial, wraps
from itertools import chain
from numbers import Number

import attr
from google.cloud.bigtable.row_filters import ValueRangeFilter, \
    ColumnQualifierRegexFilter, RowFilterChain, BlockAllFilter
import numpy as np
import pandas as pd

from . import nxtime
from .functions import get, passthrough, iformat


__all__ = ['FloatRange', 'FloatInterval', 'EmptyFloatInterval',
           'FloatSingleton', 'TimeRange', 'TimeInterval', 'EmptyTimeInterval',
           'TimeSingleton', 'Levels', 'Level', 'MultiFloatInterval',
           'MultiTimeInterval', 'float_range', 'time_range',
           'categorical_range', 'cat_range']


def nonehandler(default):
    """Conversion function decorator that handles None with a default."""
    def decorator(converter):
        @wraps(converter)
        def decorated(value):
            if value is None:
                return default
            else:
                return converter(value)
        return decorated
    return decorator

# =============================================================================
# Abstract base classes
# =============================================================================


class Range(abc.ABC):
    """Abstract base class for all Range objects."""

    @property
    def serializable(self):
        return None


class ContinuousRange(Range):
    """Abstract base class for Ranges in continuous data dimensions."""
    __zero__ = None  # zero-length value
    __empty__ = None  # placeholder for empty interval, assigned ex-post
    __multi__ = None  # placeholder for multi-interval type


class CategoricalRange(Range):
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


class Interval(ContinuousRange, Iterable):
    """For now intervals are closed on the left and open on the right."""

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
            return item >= self.lower and item < self.upper

    def sql(self, attr_):
        """Returns a sql statement fragment making attr within self."""
        if self.lower == self.__min__:
            lower = None
        else:
            lower = attr_ + " >= " + _sqlstring(self.lower)
        if self.upper == self.__max__:
            upper = None
        else:
            upper = attr_ + " < " + _sqlstring(self.upper)
        return " AND ".join([b for b in (lower, upper) if b is not None])

    def btfilter(self, attr_):
        """Returns bigtable row filter applying self's range to target cell."""
        colfilter = ColumnQualifierRegexFilter(attr_.encode('utf-8'))
        if self.lower == self.__min__:
            lower = None
        else:
            lower = str(self.lower).encode('utf-8')
        if self.upper == self.__max__:
            upper = None
        else:
            upper = str(self.upper).encode('utf-8')
        rgefilter = ValueRangeFilter(lower, upper)
        return RowFilterChain([colfilter, rgefilter])

    @classmethod
    def intersection(cls, a, b):
        """Returns an interval or None."""
        if isinstance(a, EmptyInterval):
            return cls.__empty__()
        elif isinstance(b, EmptyInterval):
            return cls.__empty__()
        lower = max([a.lower, b.lower])
        upper = min([a.upper, b.upper])
        if upper <= lower:
            return cls.__empty__()
        return cls(lower, upper)

    @classmethod
    def union(cls, a, b):
        """Returns either an interval or a multi-interval."""
        if isinstance(a, EmptyInterval):
            return b
        elif isinstance(b, EmptyInterval):
            return a
        multi = cls.__multi__([a, b])
        if len(multi) == 1:
            return multi[0]
        return multi

    def __and__(self, other):
        """Implements intersection at the instance level."""
        if isinstance(other, Singleton):
            return other if other.position in self else self.__empty__()
        elif isinstance(other, MultiInterval):
            return other.__and__(self)
        return self.intersection(self, other)

    def __or__(self, other):
        """Implements union at the instance level."""
        return self.union(self, other)

    def __attrs_post_init__(self):
        if self.lower > self.upper:
            msg = f"Cannot construct interval with lower bound " + \
                  f"{self.lower} greater than upper bound {self.upper}"
            raise ValueError(msg)

    def __repr__(self):
        return f"<{type(self).__name__} {self.lower}, {self.upper}>"


class EmptyInterval(Interval):
    lower = None
    upper = None

    def __init__(self, *args, **kwargs):
        pass

    @property
    def length(self):
        return self.__zero__

    def __contains__(self, item):
        return False

    def sql(self, attr_):
        return "1 < 0"

    def btfilter(self, attr_):
        return RowFilterChain([BlockAllFilter])

    @classmethod
    def intersection(cls, a, b):
        return cls()

    def __repr__(self):
        return iformat()(self)


class Singleton(ContinuousRange):
    """A degenerate continuous range at a single position."""

    @property
    def length(self):
        return self.__zero__

    @property
    def bounds(self):
        return (self.position, self.position)

    def sql(self, attr):
        """Returns a sql statement fragment making attr equals to self."""
        return attr + " = " + _sqlstring(self.position)

    def __and__(self, other):
        if isinstance(other, Singleton):
            return self if self == other else self.__empty__()
        else:
            return other.__and__(self)

    def __repr__(self):
        return f"<{type(self).__name__} {self.position}>"


class Levels(CategoricalRange, Set):
    """Holds a set of discrete levels."""

    def __init__(self, levels=None):
        if levels is None:
            levels = []
        self._levels = set(levels)

    def __contains__(self, item):
        return self._levels.__contains__(item)

    def __iter__(self):
        return self._levels.__iter__()

    def __len__(self):
        return self._levels.__len__()

    def __and__(self, other):
        if isinstance(other, Level):
            return other if other.level in self else Levels()
        else:
            return super().__and__(other)

    def _slice(self, slice_):
        if slice_.stop is None:
            return self
        elif slice_.start is None:
            return type(self)(l for l in self._levels if l < slice_.stop)
        else:
            return type(self)(l for l in self._levels if l < slice_.stop and
                              l >= slice_.start)

    def __getitem__(self, key):
        """Levels slicer."""
        if isinstance(key, str):
            if key in self._levels:
                return Level(key)
            else:
                raise KeyError
        elif isinstance(key, Iterable):
            return self & key
        elif isinstance(key, slice):
            return self._slice(key)

    @property
    def serializable(self):
        return list(self._levels)

    def __eq__(self, other):
        try:
            return self._levels == set(other)
        except TypeError:
            return False

    def __repr__(self):
        return f"<{type(self).__name__} {self._levels}>"

    def sql(self, attr):
        """Returns a sql statement fragment making attr within self."""
        return attr + " IN (" + ", ".join(_sqlstring(l) for l in self) + ")"


@attr.s(frozen=True, repr=False, cmp=False)
class Level(CategoricalRange):
    """Holds a single level."""
    level = attr.ib()

    def sql(self, attr):
        """Returns a sql statement fragment making attr equals to self."""
        return attr + " = " + _sqlstring(self.level)

    @property
    def serializable(self):
        return self.level

    @property
    def _levels(self):
        """Provided for unified interface with Levels."""
        return {self.level}

    def __and__(self, other):
        if isinstance(other, Level):
            return self if self == other else Levels()
        elif isinstance(other, Levels):
            return self if self.level in other else Levels()

    def __eq__(self, other):
        if isinstance(other, Level):
            return self.level == other.level
        elif isinstance(self, Levels):
            return False
        else:
            return self.level == other

    def __repr__(self):
        return f"<{type(self).__name__} {self.level!r}>"


# =============================================================================
# Continuous range classes
# =============================================================================


class FloatRange(ContinuousRange):
    __zero__ = 0.
    __min__ = float('-inf')
    __max__ = float('inf')


@attr.s(frozen=True, repr=False)
class FloatInterval(Interval, FloatRange):
    """An interval of floats."""
    lower = attr.ib(default=float('-inf'),
                    convert=nonehandler(float('-inf'))(np.float_))
    upper = attr.ib(default=float('inf'),
                    convert=nonehandler(float('inf'))(np.float_))

    @property
    def serializable(self):
        return (self.lower, self.upper)


class EmptyFloatInterval(EmptyInterval, FloatInterval):

    @property
    def serializable(self):
        return None


FloatRange.__empty__ = EmptyFloatInterval


@attr.s(frozen=True, repr=False)
class FloatSingleton(Singleton, FloatRange):
    position = attr.ib(convert=np.float_)

    @property
    def serializable(self):
        return self.position


class TimeRange(ContinuousRange):
    __zero__ = pd.Timedelta(0)
    __min__ = nxtime.MIN
    __max__ = nxtime.MAX


@attr.s(frozen=True, repr=False)
class TimeInterval(Interval, TimeRange):
    """An interval of datetimes.

    The arguments are automatically converted to pandas Timestamp if
    possible, potentially raising an error if that is not possible.
    Naive datetime values are also automatically converted to UTC.
    Finally, None is an admissible value for either lower or upper, in
    which case it will be converted to anaximander's absolute time bounds,
    currently set at Jan. 1 1970, UTC and Jan. 1 2100, UTC.
    """
    lower = attr.ib(nxtime.MIN, convert=nonehandler(nxtime.MIN)(
        partial(pd.to_datetime, utc=True)))
    upper = attr.ib(nxtime.MAX, convert=nonehandler(nxtime.MAX)(
        partial(pd.to_datetime, utc=True)))

    def tz_convert(self, tzinfo):
        return type(self)(self.lower.tz_convert(tzinfo),
                          self.upper.tz_convert(tzinfo))

    @property
    def duration(self):
        return self.upper - self.lower

    @property
    def serializable(self):
        return (str(self.lower), str(self.upper))


class EmptyTimeInterval(EmptyInterval, TimeInterval):

    @property
    def serializable(self):
        return None


TimeRange.__empty__ = EmptyTimeInterval


@attr.s(frozen=True, repr=False)
class TimeSingleton(Singleton, TimeRange):
    position = attr.ib(convert=partial(pd.to_datetime, utc=True))

    def tz_convert(self, tzinfo):
        return type(self)(self.position.tz_convert(tzinfo))

    @property
    def serializable(self):
        return str(self.position)

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


def mcompare(method):
    """Decorator for multi-interval comparison methods."""
    @wraps(method)
    def decorated(self, other):
        cls = type(self)
        if isinstance(other, Interval):
            multi = method(self, cls([other]))
        elif isinstance(other, MultiInterval):
            multi = method(self, other)
        elif isinstance(other, Singleton) and method.__name__ == '__and__':
            if any(other.position in i for i in self):
                return other
            else:
                return self.__empty__()
        else:
            raise NotImplementedError
        if len(multi) == 0:
            return self.__empty__()
        elif len(multi) == 1:
            return multi[0]
        else:
            return multi
    return decorated


class MultiInterval(MultiRange, Sequence):
    """An object representing a union of intervals."""
    tolerance = None  # Optional tolerance for interval overlap

    def __init__(self, intervals=None):
        intervals = get(intervals, [])
        self._intervals = self.normalize(*intervals)

    @property
    def intervals(self):
        return list(self._intervals)

    @classmethod
    def normalize(cls, *intervals):
        """Normalizes intervals into an ordered, non-overlapping set."""
        intervals = list(i for i in intervals
                         if not isinstance(i, EmptyInterval))
        if not intervals:
            return []
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
        return [cls.itype(lower, upper)
                for lower, upper in zip(lowers, uppers)]

    @classmethod
    def intersection(cls, a, b):
        """Returns the intersection of two MultiIntervals, as a MultiInterval.

        This method assumes that a and b are MultiInterval instances,
        and, crtically, that they have already been normalized.
        """
        if not a:
            return cls()
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
        return cls([cls.itype(l, u) for l, u in zip(lowers, uppers)])

    @classmethod
    def difference(cls, a, b):
        """Returns the difference of two MultiIntervals, as a MultiInterval.

        This method assumes that a and b are MultiInterval instances,
        and, crtically, that they have already been normalized.
        """
        if not a:
            return cls()
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
        return cls([cls.itype(l, u) for l, u in zip(lowers, uppers)])

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

    @mcompare
    def __and__(self, other):
        return self.intersection(self, other)

    @mcompare
    def __sub__(self, other):
        return self.difference(self, other)

    @mcompare
    def __or__(self, other):
        return self.union(self, other)

    def __repr__(self):
        return iformat()(self)


class MultiFloatInterval(MultiInterval, FloatRange):
    itype = FloatInterval


FloatRange.__multi__ = MultiFloatInterval


class MultiTimeInterval(MultiInterval, TimeRange):
    itype = TimeInterval
    tolerance = pd.Timedelta(microseconds=1)

    @property
    def duration(self):
        if not self:
            return pd.Timedelta(0)
        durations = pd.Series(i.length for i in self)
        return durations.sum()

    def tz_convert(self, tzinfo):
        return type(self)([i.tz_convert(tzinfo) for i in self._intervals])


TimeRange.__multi__ = MultiTimeInterval

# =============================================================================
# Helper functions
# =============================================================================


@passthrough(MultiFloatInterval, FloatInterval, FloatSingleton)
def float_range(x=None, y=None):
    """Creates or passes through a float range from one or two arguments."""
    if x is None:
        return EmptyFloatInterval()
    if y is None:
        if isinstance(x, Iterable) and not isinstance(x, str):
            return FloatInterval(*x)
        elif isinstance(x, slice):
            return FloatInterval(x.start, x.stop)
        else:
            return FloatSingleton(x)
    else:
        return FloatInterval(x, y)


@passthrough(MultiTimeInterval, TimeInterval, TimeSingleton)
def time_range(t0=None, t1=None):
    """Creates or passes through a time range from one or two arguments."""
    if t0 is None:
        return EmptyTimeInterval()
    if t1 is None:
        if isinstance(t0, Iterable) and not isinstance(t0, str):
            return TimeInterval(*t0)
        elif isinstance(t0, slice):
            return TimeInterval(t0.start, t0.stop)
        else:
            return TimeSingleton(t0)
    else:
        return TimeInterval(t0, t1)


@passthrough(Level, Levels)
def categorical_range(arg=None, *, sliced=None):
    """Creates or passes through either a Level or Levels.

    Admits a slice argument if a Levels range is provided as kwarg 'sliced'.
    """
    if arg is None:
        return Levels([])
    if isinstance(arg, Iterable) and not isinstance(arg, str):
        return Levels(arg)
    elif isinstance(arg, slice):
        if not isinstance(sliced, Levels):
            msg = "Slice argument requires a range to be sliced."
            raise TypeError(msg)
        else:
            return sliced[arg]
    return Level(arg)

cat_range = categorical_range
