#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions related to pandas.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd

# =============================================================================
# Date / time functionalities
# =============================================================================


def delta_to_offset(timedelta):
    """Converts a pd.Timedelta object to a DateOffset."""
    attrs = {'days': 'd',
             'seconds': 's',
             'microseconds': 'us',
             'nanoseconds': 'ns'}
    params = {abbr: getattr(timedelta, attr) for attr, abbr in attrs.items()}
    return pd.DateOffset(''.join(['{0}{1}'.format(v, k)
                                  for k, v in params.items() if v != 0]))
