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

import os
import re

import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.utilities import nxrange as rge
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl, records as rec


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
FEATURELOG_INVALID_PATH = os.path.join(TEST_DATA_DIR, 'featurelog_invalid.csv')

FEATURE_IDS = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
FEATURE_TME = ['2016-9-14 10:00', '2016-9-14 10:05']

DUMP_PATH = os.path.join(TEST_DATA_DIR, 'dump.csv')

MAC_PATTERN = re.compile('^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')


# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture(scope="module")
def featurelog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(FEATURELOG_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe

LOG = featurelog()
LOG['datetime'] = pd.to_datetime(LOG.datetime, utc=True)
LOG.set_index(['id', 'datetime'], inplace=True)
LOG.sort_index(inplace=True)


@pytest.fixture(scope="module")
def featurelog_partial():
    """Returns a dataframe with missing columns."""
    dataframe = pd.read_csv(FEATURELOG_PARTIAL_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_no_device():
    """Returns a dataframe missing a key column."""
    dataframe = pd.read_csv(FEATURELOG_NO_DEVICE_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_shuffled():
    """Returns a dataframe with columns shuffled."""
    dataframe = pd.read_csv(FEATURELOG_SHUFFLED_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_superfluous():
    """Returns a dataframe with an extra column not in the schema."""
    dataframe = pd.read_csv(FEATURELOG_SUPERFLUOUS_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_invalid():
    """Returns a dataframe with invalid values."""
    dataframe = pd.read_csv(FEATURELOG_INVALID_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe


def mac_validator(record, column, value):
    if MAC_PATTERN.match(value):
        return True
    else:
        return f"{value} does not match pattern {MAC_PATTERN.pattern}"


class FeatureSchema(sch.SampleLogsSchema):
    id = cln.String(index='nominal', validate=mac_validator)
    datetime = cln.DateTime('UTC', index='sequential')
    Feature_Value_0 = cln.Float()
    Feature_Value_1 = cln.Float()
    Feature_Value_2 = cln.Float()
    Feature_Value_3 = cln.Float()
    Feature_Value_4 = cln.Float()
    Feature_Value_5 = cln.Float()

    @id.validator
    def second_validation(record, column, value):
        return mac_validator(record, column, value)


def test_logs(featurelog, featurelog_partial, featurelog_no_device,
              featurelog_shuffled, featurelog_superfluous,
              featurelog_invalid):
    log = dtl.DataLog(featurelog, schema=FeatureSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    assert log.data.equals(LOG)
    log = dtl.DataLog(featurelog_partial,
                      schema=FeatureSchema('Feature_Value_0'),
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    assert log.data.equals(LOG[['Feature_Value_0']])
    with pytest.raises(dtl.ConformityError):
        dtl.DataLog(featurelog_no_device, schema=FeatureSchema,
                    id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    log = dtl.DataLog(featurelog_shuffled, schema=FeatureSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    assert log.data.equals(LOG)
    log = dtl.DataLog(featurelog_superfluous, schema=FeatureSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    assert log.data.equals(LOG)
    log = dtl.DataLog(featurelog_invalid, schema=FeatureSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    assert not log.validate()


def test_json_round_trip(featurelog):
    log = dtl.DataLog(featurelog, schema=FeatureSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    log = dtl.DataLog.json_loads(log.json_dumps())
    assert log.data.equals(LOG)


def test_slicing(featurelog):
    log = dtl.DataLog(featurelog, schema=FeatureSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    l0 = log['68:9E:19:07:DE:C3']
    assert isinstance(l0, dtl.DataSequence)
    assert len(l0.data) == 3
    assert l0.id_range == '68:9E:19:07:DE:C3'
    assert l0.dt_range == rge.time_range(*FEATURE_TME)
    l1 = l0['2016-09-14 10:00:00':'2016-9-14 10:01:00']
    assert isinstance(l1, dtl.DataSequence)
    assert len(l1.data) == 2
    assert l1.dt_range.upper == pd.to_datetime('2016-9-14 10:01', utc=True)
    l2 = log[['68:9E:19:07:DE:C3'], '2016-09-14 10:00:00':'2016-9-14 10:01:00']
    assert isinstance(l2, dtl.DataLog)
    assert l2.id_range == {'68:9E:19:07:DE:C3'}
    assert len(l2.data) == 2
    assert not l1.data.equals(l2.data)  # simple index vs. multi-index
    assert l1.tabulated.equals(l2.tabulated)
    l3 = log[slice(None), '2016-09-14 10:00:00':'2016-9-14 10:01:00']
    assert isinstance(l3, dtl.DataLog)
    assert l3.id_range == FEATURE_IDS
    assert l3.dt_range.upper == pd.to_datetime('2016-9-14 10:01', utc=True)
    assert len(l3.data) == 7
    l4 = log[slice(None), '2016-09-14 10:00:29.500']
    assert isinstance(l4, dtl.DataArray)
    assert l4.id_range == FEATURE_IDS
    assert l4.dt_range == rge.TimeSingleton('2016-09-14 10:00:29.500')
    assert len(l4.data) == 1
    l5 = log[slice(None), slice(None)]
    assert l5 == log
    with pytest.raises(KeyError):
        log[slice(None), '2016-09-14 10:00:29.600']
    l6 = log[slice(None), '2016-09-14 10:10:00':'2016-9-14 10:11:00']
    assert isinstance(l6, dtl.DataLog)
    assert l6.id_range == FEATURE_IDS
    assert l6.dt_range == rge.EmptyTimeInterval()
    assert l6.empty
    l7 = log['68':'78']
    assert isinstance(l7, dtl.DataLog)
    assert l7.id_range == {'68:9E:19:07:DE:C3'}
    assert l7.dt_range == rge.time_range(*FEATURE_TME)
    assert l7.tabulated.equals(l0.tabulated)
    l8 = log['90':'xx', slice(None)]
    assert isinstance(l8, dtl.DataLog)
    assert l8.id_range == {}
    assert l8.dt_range == rge.time_range(*FEATURE_TME)
    assert l8.empty
    r = log['68:9E:19:07:DE:C3', '2016-09-14 10:01:32.990000+00:00']
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:01:32.990', utc=True)
    r = l1['2016-09-14 10:00:29.500']
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:00:29.500', utc=True)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
