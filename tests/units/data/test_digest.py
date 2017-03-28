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

import pytest

import anaximander as nx
from anaximander.utilities.nxtime import datetime
from anaximander.data import annotations as ant, schema, tract
from anaximander.data import digest as dig
from anaximander.data import NxFloat
from anaximander.data.fields import Timestamp, Scalar, ReStr

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
FEATURELOG_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')
MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
mac = '88:4A:EA:69:DF:A2'

# =============================================================================
# Test Cases
# =============================================================================


class MyMarker(ant.Marker):
    red = ant.shade(color='red')


class MyHighlighter(ant.Highlighter):
    yellow = ant.shade(color='yellow')


class ThresholdMarkSurvey(dig.MarkSurvey):
    markertype = MyMarker

    def __marks__(self, series, threshold=0):
        subset = series[series >= threshold]
        return [self.mark('red', loc) for loc in subset.index]


class ThresholdHighlightSurvey(dig.HighlightSurvey):
    highlightertype = MyHighlighter

    def __highlights__(self, series, threshold=0):
        above = series >= threshold
        groups = above.diff().cumsum()
        groups.iloc[0] = 0
        grouped = series[above].groupby(groups)
        if not list(grouped):
            return []
        _, clusters = zip(*grouped)
        return [self.highlight('yellow', c.index[0], c.index[-1])
                for c in clusters]


@tract
class Feature(schema.Schema):
    device = ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = Timestamp(key=True, sequential=True)
    Feature_Value_0 = Scalar(NxFloat)


@tract
class Turefea(schema.Schema):
    device = ReStr(key=True, pattern=MAC_PATTERN)    
    Feature_Value_0 = Scalar(NxFloat, key=True, sequential=True)
    timestamp = Timestamp()


@pytest.fixture(scope="module")
def featurelog():
    """Returns a feature dataframe."""
    return Feature.Frame.from_csv(FEATURELOG_PATH)[mac]

@pytest.fixture(scope="module")
def turefealog():
    """Returns a turefea dataframe."""
    return Turefea.Frame.from_csv(FEATURELOG_PATH)[mac]


def test_instantiation(featurelog, turefealog):
    d = dig.MarkDigest(featurelog, MyMarker)
    assert type(d) == dig.TimeMarkDigest
    d = dig.MarkDigest(turefealog, MyMarker)
    assert type(d) == dig.FloatMarkDigest
    d = dig.MarkDigest(Feature.Sequence(), MyMarker)
    assert type(d) == dig.EmptyMarkDigest
    h = dig.HighlightDigest(featurelog, MyHighlighter)
    assert type(h) == dig.TimeHighlightDigest
    h = dig.HighlightDigest(turefealog, MyHighlighter)
    assert type(h) == dig.FloatHighlightDigest
    h = dig.HighlightDigest(Feature.Sequence(), MyHighlighter)
    assert type(h) == dig.EmptyHighlightDigest    


def test_mark_survey(featurelog, turefealog):
    survey = ThresholdMarkSurvey(featurelog, 'Feature_Value_0', threshold=500)
    assert len(survey()) == 2
    survey = ThresholdMarkSurvey(turefealog, 'Feature_Value_0', threshold=500)
    assert len(survey()) == 2


def test_highlight_survey(featurelog, turefealog):
    survey = ThresholdHighlightSurvey(featurelog, 2, threshold=10)
    assert len(survey()) == 4
    dt_threshold = datetime.min
    survey = ThresholdHighlightSurvey(turefealog, 2, threshold=dt_threshold)
    assert len(survey()) == 2


if __name__ == '__main__':
    pytest.main([__file__])
