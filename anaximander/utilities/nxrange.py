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
from collections import Iterable
from collections.abc import Set

from .nxtime import datetime
from . import nxattr


__all__ = []

# =============================================================================
# Utilities
# =============================================================================

class Range(abc.ABC):
    """Abstract base class for all Range objects."""
    pass


class ContinuousRange(Range):
    """Abstract base class for Ranges in continuous data dimensions."""
    pass


class DiscreteRange(Range):
    """Abstract base class for Ranges in discrete data dimensions.

    This base class can be called upon and will automatically select the
    proper subclass (Levels or Level) based on the type of input.
    If the input is an iterable, then a Levels instance will be created.
    This obviously limits levels to scalar values. This limitation may
    be addressed in the future if needed.
    """

    def __new__(cls, arg):
        if isinstance(arg, Iterable) and not isinstance(arg, str):
            return Levels(arg)
        else:
            return Level(arg)


@nxattr.s
class Interval(ContinuousRange):
    lower = nxattr.ib()
    upper = nxattr.ib()

    @property
    def length(self):
        return self.upper - self.lower

    def __contains__(self, item):
        if isinstance(item, Interval):
            return item.lower >= self.lower and item.upper <= self.upper
        else:
            return item >= self.lower and item <= self.upper


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


class Levels(DiscreteRange, Set):

    def __new__(cls, arg):
        return Range.__new__(cls)

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

    def __new__(cls, arg):
        return Range.__new__(cls)

    def __init__(self, level):
        self._level = level

    def __repr__(self):
        return "Level({0})".format(repr(self._level))
