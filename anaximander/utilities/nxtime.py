#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light wrapper around pandas datetime functionalities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import pytz
import pandas as pd

__all__ = ['datetime', 'pydatetime', 'timestamp', 'timezone', 'UTC']

timezone = pytz.timezone
UTC = pytz.UTC

# =============================================================================
# Basic functionalities
# =============================================================================


MIN = pd.Timestamp('1970-1-1', tz=UTC)
MAX = pd.Timestamp('2100-1-1', tz=UTC)

PYMIN = MIN.to_pydatetime()
PYMAX = MAX.to_pydatetime()

MIN_TIMESTAMP = MIN.timestamp()
MAX_TIMESTAMP = MAX.timestamp()


def datetime(value):
    """Equivalent to pd.Timestamp, but converts naive datetime to UTC."""
    timestamp = pd.to_datetime(value)
    try:
        if not timestamp.tz:
            return timestamp.tz_localize(UTC)
    except AttributeError:
        pass
    return timestamp

datetime.min = MIN
datetime.max = MAX


def now():
    return pd.Timestamp.utcnow()


def naify(dt):
    """Returns a tz_naive pd.Timestamp, after conversion to UTC if needed."""
    timestamp = pd.to_datetime(dt)
    if not timestamp.tz:
        return timestamp
    else:
        return timestamp.tz_convert(UTC).tz_localize(None)


def pydatetime(value):
    """Returns Python datetime from any value that datetime can interpret."""
    return datetime(value).to_pydatetime(warn=False)

pydatetime.min = PYMIN
pydatetime.max = PYMAX


def timestamp(value):
    """Returns a UNIX timestamp from any value that datetime can interpret."""
    return datetime(value).timestamp()

timestamp.min = MIN_TIMESTAMP
timestamp.max = MAX_TIMESTAMP

# =============================================================================
# Pandas extensions
# =============================================================================


def tz_aware(data):
    """Makes a series or dataframe tz-aware and returns the result.

    Naive datetimes are converted to UTC.
    """
    data = data.copy()
    try:
        index = data.index
    except AttributeError:
        pass
    else:
        if index.dtype.kind == 'M':
            if index.tz is None:
                data = data.tz_localize('utc')
        elif index.nlevels > 1:
            for i, lv in enumerate(index.levels):
                if lv.dtype.kind == 'M':
                    if lv.tz is None:
                        data = data.tz_localize('utc', axis=i)
    if isinstance(data, pd.Series):
        if data.dtype.kind == 'M':
            data = data.apply(datetime)
    elif isinstance(data, pd.DataFrame):
        for col, dt in dict(data.dtypes).items():
            if dt.kind == 'M':
                data[col] = data[col].apply(datetime)
    elif isinstance(data, pd.Index):
        return tz_aware(pd.Series(index=data)).index
    return data


def tz_naive(data):
    """Makes a series or dataframe datetime fields naive.

    Non-naive datetimes are first converted to UTC.
    """
    data = data.copy()
    try:
        index = data.index
    except AttributeError:
        pass
    else:
        if index.dtype.kind == 'M':
            if index.tz is not None:
                data = data.tz_convert('utc').tz_localize(None)
        elif index.nlevels > 1:
            for i, lv in enumerate(index.levels):
                if lv.dtype.kind == 'M':
                    if lv.tz is None:
                        data = data.tz_convert('utc', axis=i)
                        data = data.tz_localize(None, axis=i)
    if isinstance(data, pd.Series):
        if data.dtype.kind == 'M':
            data = data.apply(naify)
    elif isinstance(data, pd.DataFrame):
        for col, dt in dict(data.dtypes).items():
            if dt.kind == 'M':
                data[col] = data[col].apply(naify)
    elif isinstance(data, pd.Index):
        return tz_naive(pd.Series(index=data)).index
    return data


def dates(data, tz='utc'):
    """Returns a set of local dates from a tz-aware time-indexed data."""
    if isinstance(data, pd.Index):
        index = data
    else:
        index = data.index
    if index.tz is None:
        msg = "The dates function requires a tz-aware index, not {0}."
        raise ValueError(msg.format(index))
    return set(index.tz_convert(tz).date)
