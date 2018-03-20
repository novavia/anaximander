#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxtime.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import os.path

import pandas as pd
import pytest
import pytz

import anaximander as nx
from anaximander.utilities.nxtime import datetime, naify, tz_aware, tz_naive


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
FEATURELOG_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture(scope="module")
def featurelog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(FEATURELOG_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


def test_datetime():
    naive = '2017-03-23 00:00:00'
    assert datetime(naive) == pd.Timestamp('2017-03-23 00:00:00 UTC')
    aware = '2017-03-23 00:00:00 UTC'
    assert datetime(aware) == pd.Timestamp('2017-03-23 00:00:00 UTC')
    aware = '2017-03-23 00:00:00-0700'
    assert datetime(aware) == pd.Timestamp('2017-03-23 00:00:00-0700')


def test_naify():
    naive = '2017-03-23 00:00:00'
    assert naify(naive) == pd.Timestamp('2017-03-23 00:00:00')
    aware = '2017-03-23 00:00:00 UTC'
    assert naify(aware) == pd.Timestamp('2017-03-23 00:00:00')
    aware = '2017-03-23 00:00:00-0700'
    assert naify(aware) == pd.Timestamp('2017-03-23 07:00:00')


def test_tz_aware(featurelog):
    dataframe = tz_aware(featurelog)
    assert dataframe.timestamp[0].tz == pytz.UTC
    dataframe = tz_naive(dataframe)
    assert dataframe.timestamp[0].tz is None


if __name__ == '__main__':
    pytest.main([__file__])
