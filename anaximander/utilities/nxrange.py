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

from .nxtime import datetime
from . import nxattr
from .functions import passthrough


__all__ = ['float_interval', 'time_interval', 'levels']

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


@nxattr.s
class Interval(ContinuousRange):
    lower = nxattr.ib()
    upper = nxattr.ib()

    @property
    def length(self):
        return self.upper - self.lower

    @property
    def bounds(self):
        return (self.lower, self.upper)

    def __contains__(self, item):
        if isinstance(item, Interval):
            return item.lower >= self.lower and item.upper <= self.upper
        else:
            return item >= self.lower and item <= self.upper



def _lower_float_convert(value):
    if value is None:
        return float('-inf')
    else:
        return float(value)


def _upper_float_convert(value):
    if value is None:
        return float('-inf')
    else:
        return float(value)


@nxattr.s(inherit=False)
class FloatInterval(Interval):
    """An interval of floats."""
    lower = nxattr.ib(convert=_lower_float_convert)
    upper = nxattr.ib(convert=_upper_float_convert)    


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


@nxattr.s(inherit=False)
class TimeInterval(Interval):
    """An interval of datetimes.

    The arguments are automatically converted to pandas Timestamp if
    possible, potentially raising an error if that is not possible.
    Naive datetime values are also automatically converted to UTC.
    Finally, None is an admissible value for either lower or upper, in
    which case it will be converted to anaximander's absolute time bounds,
    currently set at Jan. 1 1970, UTC and Jan. 1 2100, UTC.
    """
    lower = nxattr.ib(convert=_lower_time_convert)
    upper = nxattr.ib(convert=_upper_time_convert)


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


@nxattr.s(inherit=False)
class StringInterval(Interval):
    """An interval of strings."""
    lower = nxattr.ib(convert=_lower_string_convert)
    upper = nxattr.ib(convert=_upper_string_convert)


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

    def __repr__(self):
        return "Levels({0})".format(repr(self._levels))


class Level(DiscreteRange):
    """Holds a single level."""

    def __init__(self, level):
        self._level = level

    def __eq__(self, other):
        return self._level.__eq__(other)

    def __repr__(self):
        return "Level({0})".format(repr(self._level))

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
