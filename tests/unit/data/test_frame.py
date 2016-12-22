#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for frame.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from collections import OrderedDict
import os
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

import anaximander as nx
from anaximander.data import fields, frame, schema


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
FEATURELOG_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')
FEATURELOG_PARTIAL_PATH = os.path.join(TEST_DATA_DIR, 'featurelog_partial.csv')
FEATURELOG_NO_DEVICE_PATH = os.path.join(TEST_DATA_DIR,
                                         'featurelog_no_device.csv')
FEATURELOG_SHUFFLED_PATH = os.path.join(TEST_DATA_DIR,
                                        'featurelog_shuffled.csv')
FEATURELOG_SUPERFLUOUS_PATH = os.path.join(TEST_DATA_DIR,
                                           'featurelog_superfluous.csv')

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def featurelog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(FEATURELOG_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


@pytest.fixture
def featurelog_partial():
    """Returns a dataframe with missing columns."""
    dataframe = pd.read_csv(FEATURELOG_PARTIAL_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


@pytest.fixture
def featurelog_no_device():
    """Returns a dataframe missing a key column."""
    dataframe = pd.read_csv(FEATURELOG_NO_DEVICE_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


@pytest.fixture
def featurelog_shuffled():
    """Returns a dataframe with columns shuffled."""
    dataframe = pd.read_csv(FEATURELOG_SHUFFLED_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


@pytest.fixture
def featurelog_superfluous():
    """Returns a dataframe with an extra column not in the schema."""
    dataframe = pd.read_csv(FEATURELOG_SUPERFLUOUS_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


class FeatureSchema(schema.Schema):
    device = fields.Str(key=True)
    timestamp = fields.DateTime(key=True, sequential=True)
    Feature_Value_0 = fields.Float()
    Feature_Value_1 = fields.Float()
    Feature_Value_2 = fields.Float()
    Feature_Value_3 = fields.Float()
    Feature_Value_4 = fields.Float()
    Feature_Value_5 = fields.Float()


class FeatureFrame(frame.Frame):
    pass

FeatureFrame.schema = FeatureSchema


def test_dtypes():
    """Tests the dtypes function."""
    spec = OrderedDict([('device', np.dtype('object')),
                        ('timestamp', np.dtype('datetime64[ns]')),
                        ('Feature_Value_0', np.dtype('float')),
                        ('Feature_Value_1', np.dtype('float')),
                        ('Feature_Value_2', np.dtype('float')),
                        ('Feature_Value_3', np.dtype('float')),
                        ('Feature_Value_4', np.dtype('float')),
                        ('Feature_Value_5', np.dtype('float'))])
    assert frame.dtypes(FeatureSchema) == spec


class TestFrame(TestCase):

    def test_cast(self):
        log = featurelog()
        native_log = log.astype({'timestamp': np.dtype('object')})
        cast = FeatureFrame.cast(native_log)
        # Casts converts the timestamp back to datetime
        assert cast.equals(log)
        assert not cast.equals(native_log)

    def test_cast_partial(self):
        """Tests with a dataframe missing non-essential columns."""
        log = featurelog_partial()
        cast = FeatureFrame.cast(log)
        assert cast.equals(log)

    def test_cast_no_device(self):
        """Tests with a dataframe missing key column 'device'."""
        log = featurelog_no_device()
        with pytest.raises(frame.ConformityError):
            FeatureFrame.cast(log)

    def test_cast_shuffled(self):
        """Tests with a dataframe whose columns are out of sequence."""
        log = featurelog_shuffled()
        cast = FeatureFrame.cast(log)
        assert frame.dtypes(FeatureSchema) == OrderedDict(cast.dtypes)

    def test_cast_superfluous(self):
        """Tests with a dataframe with a superfluous column."""
        log = featurelog_superfluous()
        cast = FeatureFrame.cast(log)
        assert len(log.columns) == len(cast.columns) + 1
        assert frame.dtypes(FeatureSchema) == OrderedDict(cast.dtypes)

    def test_init(self):
        log = featurelog()
        frame = FeatureFrame(log)
        assert frame.data.equals(log)

    def test_validate(self):
        log = featurelog()
        frame = FeatureFrame(log, validate=True)
        assert frame.data.equals(log)

if __name__ == '__main__':
    pytest.main([__file__])
