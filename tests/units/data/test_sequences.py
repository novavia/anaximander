#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for data.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import os.path

import pandas as pd
import pytest

import anaximander as nx
from anaximander.utilities import functions as fun
from anaximander.utilities.nxtime import datetime, MIN, MAX
from anaximander.data.sequences import SampleSequence, EventSequence, \
    PhaseSequence, PeriodSequence, Thresholder, SessionMaker

LOWER = datetime('2018-1-1 00:00:00')
UPPER = datetime('2018-1-1 00:01:00')

# =============================================================================
# Test Cases
# =============================================================================


def samples(*highs, lower=None, upper=None):
    """Generates a SampleSequence for testing different inputs.

    Params:
        *highs: any number of integers between 0 and 60.
    Returns:
        A SampleSequence object.

    The SampleSequence spans exactly one minute, with one sample every
    second defaulting to a value of 0. The collection of highs changes
    the value to 1 for each second reference it contains. For instance
    if highs = [5, 6, 7], then the samples at 00:00:05, 00:00:06 and 00:00:07
    will be 1.
    """
    lower = fun.get(lower, LOWER)
    upper = fun.get(upper, UPPER)
    index = pd.DatetimeIndex(start=lower, freq='s', periods=61)
    series = pd.Series(0, index=index)
    series.iloc[list(highs)] = 1
    df = series.to_frame('samples')
    df['timestamp'] = df.index
    return SampleSequence(df, lower=lower, upper=upper)


def test_thresholder():
    sequence = samples(20, 40)
    thresholder = Thresholder(sequence, 1)
    output = thresholder()
    assert len(output) == 2
    assert output.lower == LOWER


def test_sessionmaker():

    def test(highs, session_count, session_duration, lower, upper, onstate,
             sample_lower=None, sample_upper=None,
             max_gap='5s', min_span=None):
        sequence = samples(*highs, lower=sample_lower, upper=sample_upper)
        peaks = Thresholder(sequence, 1)()
        states = SessionMaker(peaks, max_gap=max_gap, min_span=min_span)()
        sessions = states.phases
        assert len(sessions) == session_count
        session_duration = pd.Timedelta(session_duration)
        slengths = pd.Series(p.length for p in sessions)
        assert slengths.sum() == session_duration
        assert states.lower == datetime(lower)
        assert states.upper == datetime(upper)

    # Two single-event clusters
    test([20, 30], 2, 0, LOWER, UPPER, 'blank')
    # Two continuous clusters
    test(list(range(10, 21)) + list(range(30, 41)),
         2, '20s', LOWER, UPPER, 'blank')
    # An event close to the lower bound
    test([2] + list(range(10, 21)),
         2, '10s', '2018-1-1 00:00:02', UPPER, 'session')
    # One event close to the lower bound and one close to the upper bound
    test([2, 57], 2, 0, '2018-1-1 00:00:02', '2018-1-1 00:00:57', 'session')
    # Event range starts within a long cluster
    test(list(range(0, 11)), 1, '10s', LOWER, UPPER, 'session')
    # The event lower bound is higher than its smallest index
    test(list(range(10, 21)), 1, '10s', '2018-1-1 00:00:05', UPPER,
         'blank', sample_lower='2018-1-1 00:00:05')
    # Event upper bound lower than its highest index and close to high event
    # Also throwing some events in outside the events certification bounds
    test(list(range(10, 21)) + list(range(30, 41)), 1, '10s',
         LOWER, '2018-1-1 00:00:20', 'blank', sample_upper='2018-1-1 00:00:24')
    # No event
    test([], 0, 0, LOWER, UPPER, 'blank')
    # No event, max gap larger than the event window
    test([], 0, 0, MIN, MAX, 'blank', max_gap='70s')
    # Adding min span, with non-qualifying event close to upper bound
    test(list(range(10, 21)) + [30, 31, 32, 57, 58], 1, '10s',
         LOWER, '2018-1-1 00:00:57', 'blank', min_span='5s')
    # Non-qualifying event near lower bound, and non-contiguous cluster
    test(list(range(10, 21)) + [3, 4, 25, 53], 1, '15s',
         '2018-1-1 00:00:04', UPPER, 'blank', min_span='5s')


if __name__ == '__main__':
    pytest.main([__file__])
