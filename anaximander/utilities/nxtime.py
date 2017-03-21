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
from pandas import Timestamp

__all__ = ['datetime']


# =============================================================================
# Time constants
# =============================================================================


MIN = Timestamp('1970-1-1', tz=pytz.utc)
MAX = Timestamp('2100-1-1', tz=pytz.utc)

PyMIN = MIN.to_pydatetime()
PyMAX = MAX.to_pydatetime()

MIN_TIMESTAMP = MIN.timestamp()
MAX_TIMESTAMP = MAX.timestamp()


def datetime(value):
    """Equivalent to pd.Timestamp, but converts naive datetime to UTC."""
    timestamp = Timestamp(value)
    if not timestamp.tz:
        return timestamp.tz_localize(pytz.UTC)
    else:
        return timestamp

datetime.min = MIN
datetime.max = MAX
