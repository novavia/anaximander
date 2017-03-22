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
# Time constants
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
    if not timestamp.tz:
        return timestamp.tz_localize(UTC)
    else:
        return timestamp

datetime.min = MIN
datetime.max = MAX


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
