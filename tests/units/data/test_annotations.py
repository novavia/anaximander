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
from anaximander.utilities.nxtime import datetime
from anaximander.utilities.nxrange import time_interval
from anaximander.data import annotations as ant, schema, tract
from anaximander.data import NxFloat
from anaximander.data.fields import DateTime, Timestamp, Float, Scalar, ReStr

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
FEATURELOG_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')
MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def markertype():

    class MyMarker(ant.Marker):
        red = ant.shade(color='red')
    
    return MyMarker


@tract
class Feature(schema.Schema):
    device = ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = Timestamp(key=True, sequential=True)
    Feature_Value_0 = Scalar(NxFloat)
    Feature_Value_1 = Scalar(NxFloat)
    Feature_Value_2 = Scalar(NxFloat)
    Feature_Value_3 = Scalar(NxFloat)
    Feature_Value_4 = Scalar(NxFloat)
    Feature_Value_5 = Scalar(NxFloat)


@pytest.fixture(scope="module")
def featurelog():
    """Returns a dataframe."""
    return Feature.Frame.from_csv(FEATURELOG_PATH)


def test_instantiation(markertype):
    green = markertype('green')
    assert list(markertype.shades.keys()) == ['blank', 'red', 'green']
    assert markertype('green') == green


def test_declaration():

    class MyMarker(ant.Marker):
        red = ant.shade(color='red')
        green = ant.shade(color='green')
        
    assert list(MyMarker.shades.keys()) == ['blank', 'red', 'green']
    assert MyMarker('red').plargs == {'color': 'red'}


def test_interval():
    lower_time = '2017-3-20'
    upper_time = '2017-3-21'
    lower_float = 0.
    upper_float = 1.
    interval = ant.interval(lower_time, upper_time, ref=Timestamp)
    assert interval.lower == datetime('2017-3-20')
    interval = ant.interval(lower_time, upper_time, ref=Timestamp())
    assert interval.lower == datetime('2017-3-20')
    interval = ant.interval(lower_time, upper_time, ref=DateTime)
    assert interval.lower == datetime('2017-3-20')
    interval = ant.interval(lower_time, upper_time, ref=DateTime())
    assert interval.lower == datetime('2017-3-20')
    interval = ant.interval(lower_time, upper_time, ref='M')
    assert interval.lower == datetime('2017-3-20')
    interval = ant.interval(lower_time, upper_time, ref=pd.DatetimeIndex)
    assert interval.lower == datetime('2017-3-20')
    index = pd.DatetimeIndex((lower_time, upper_time))
    interval = ant.interval(lower_time, upper_time, ref=index)
    assert interval.lower == datetime('2017-3-20')
    interval = ant.interval(lower_float, upper_float, ref=Float)
    assert interval.lower == 0
    interval = ant.interval(lower_float, upper_float, ref=Float())
    assert interval.lower == 0    
    interval = ant.interval(lower_float, upper_float, ref=Scalar(NxFloat))
    assert interval.lower == 0
    interval = ant.interval(lower_float, upper_float, ref='f')
    assert interval.lower == 0
    interval = ant.interval(lower_float, upper_float, ref=pd.Float64Index)
    assert interval.lower == 0
    index = pd.Index((lower_float, upper_float))
    interval = ant.interval(lower_float, upper_float, ref=index)
    assert interval.lower == 0
    interval = ant.interval(lower_float, upper_float, ref=NxFloat)
    assert interval.lower == 0
    interval = ant.interval(lower_float, upper_float, ref=NxFloat(3))
    assert interval.lower == 0
    with pytest.raises(ValueError):
        interval = ant.interval(lower_time, upper_time, ref=NxFloat)
    with pytest.raises(ValueError):
        interval = ant.interval(lower_time, upper_time, ref='ref')
    with pytest.raises(TypeError):
        interval = ant.interval(lower_time, upper_time)        


def test_marker(featurelog, markertype):
    marker = markertype('red')
    mark = ant.Mark(featurelog, marker, '2016-9-14 10:03')    
    assert mark.location == datetime('2016-9-14 10:03')
    assert type(mark.data) == Feature.Record
    assert mark.dataloc == datetime('2016-09-14 10:02:27.800')


def test_highlight(featurelog, markertype):
    marker = markertype('red')
    highlight = ant.Highlight(featurelog, marker, '2016-9-14 10:03')
    assert highlight.interval == time_interval('2016-9-14 10:03')
    assert type(highlight.data) == type(featurelog)
    assert len(highlight.data) == 5
    series = featurelog.Feature_Value_0()
    highlight = ant.Highlight(series, marker, '2016-9-14 10:03')
    assert type(highlight.data) == type(series)
    assert len(highlight.data) == 5


if __name__ == '__main__':
    pytest.main([__file__])
