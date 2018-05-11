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

import numpy as np
import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.utilities import nxrange as rge
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl, records as rec, plot


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
SAMPLES_PATH = os.path.join(TEST_DATA_DIR, 'samples.csv')
EVENTS_PATH = os.path.join(TEST_DATA_DIR, 'events.csv')
STATES_PATH = os.path.join(TEST_DATA_DIR, 'states.csv')
SESSIONS_PATH = os.path.join(TEST_DATA_DIR, 'sessions.csv')
PERIODS_PATH = os.path.join(TEST_DATA_DIR, 'periods.csv')

SAMPLES = pd.read_csv(SAMPLES_PATH)
EVENTS = pd.read_csv(EVENTS_PATH)
STATES = pd.read_csv(STATES_PATH)
SESSIONS = pd.read_csv(SESSIONS_PATH)
PERIODS = pd.read_csv(PERIODS_PATH)


FEATURE_IDS = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
FEATURE_TME = ['2016-9-14 10:00', '2016-9-14 10:05']

IDS = ['88:4A:EA:69:35:BD', '88:4A:EA:69:38:1A']
SAMPLES_TME = ['2018-4-15 00:15:00', '2018-4-15 00:15:10']
EVENTS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
STATES_TME = ['2018-4-15 00:16:00', '2018-4-15 00:17:00']
SESSIONS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
PERIODS_TME = ['2018-4-14 00:00:00', '2018-4-15 00:15:00']

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


class FeatureSchema(sch.SampleLogsSchema):
    id = cln.String(index='nominal', validate=cln.RegExValidator(MAC_PATTERN))
    datetime = cln.DateTime('UTC', index='sequential')
    Feature_Value_0 = cln.Float()
    Feature_Value_1 = cln.Float()
    Feature_Value_2 = cln.Float()
    Feature_Value_3 = cln.Float()
    Feature_Value_4 = cln.Float()
    Feature_Value_5 = cln.Float()


class SampleSchema(sch.SampleLogsSchema):
    id = cln.String(index='nominal')
    datetime = cln.DateTime(tz='UTC', index='sequential')
    accel_energy_512 = cln.Float()
    temperature = cln.Float()
    charge = cln.Integer()

    @id.validator
    def mac_validator(record, column, value):
        if MAC_PATTERN.match(value):
            return True
        else:
            return f"{value} does not match pattern {MAC_PATTERN.pattern}"


class EventSchema(sch.EventLogsSchema):
    label = cln.EventLabel(('heartbeat',))
    message = cln.Text()
    latency = cln.Float()


class StateSchema(sch.StateLogsSchema):
    label = cln.StateLabel(('Loading', 'Executing'))


class SessionSchema(sch.SessionLogsSchema):
    part_id = cln.Integer()


class PeriodSchema(sch.PeriodLogsSchema):
    freq = '5T'
    duration = cln.Float()
    data_coverage = cln.Percentage()
    part_count = cln.Integer()
    avg_cycle = cln.Float()
    oee = cln.Percentage()
    quota = cln.Bool()


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
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME,
                      validate=True)
    assert log() == log
    assert len(log('Feature_Value_0').columns) == 3
    with pytest.raises(ValueError):
        log('xyz')
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
    l6 = log['2016-09-14 10:10:00':'2016-9-14 10:11:00']
    assert isinstance(l6, dtl.DataLog)
    assert l6.id_range == FEATURE_IDS
    assert l6.dt_range == rge.EmptyTimeInterval()
    assert l6.empty
    r = log['68:9E:19:07:DE:C3', '2016-09-14 10:01:32.990000+00:00']
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:01:32.990', utc=True)
    r = l1['2016-09-14 10:00:29.500']
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:00:29.500', utc=True)
    r = log[0]
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:00:27.800', utc=True)
    r = l1[-1]
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:00:29.500', utc=True)
    r = l4[0]
    assert isinstance(r, rec.SampleRecord)
    assert r.id == '68:9E:19:07:DE:C3'
    assert r.datetime == pd.to_datetime('2016-09-14 10:00:29.500', utc=True)
    log_from_records = dtl.DataLog.from_records(list(log),
                                                id_range=FEATURE_IDS,
                                                dt_range=FEATURE_TME)
    assert log_from_records == log
    with pytest.raises(KeyError):
        log['xyz']
    with pytest.raises(KeyError):
        l1['xyz']
    assert log['2016-9-15':'2016-9-15'].empty
    assert l1['2016-9-15':'2016-9-15'].empty
    assert not l1['2016-9-15':'2018-4-15'].dt_range
    assert l1['2016-9-14 10:00:00':'2016-9-14 10:00:15'].empty
    assert l1['2016-9-14 10:00:00':'2016-9-14 10:00:15'].dt_range


def test_samples():
    log = dtl.DataLog(SAMPLES, schema=SampleSchema, id_range=IDS,
                      dt_range=SAMPLES_TME)
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    SAMPLES.loc[0, 'charge'] = None
    log = dtl.DataLog(SAMPLES, schema=SampleSchema, id_range=IDS,
                      dt_range=SAMPLES_TME)
    assert log.data.charge.iloc[0] == 0
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    r = log[0]
    assert r.data['charge'] == 0


def test_events():
    log = dtl.DataLog(EVENTS, schema=EventSchema, id_range=IDS,
                      dt_range=EVENTS_TME)
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    EVENTS.loc[0, 'label'] = None
    log = dtl.DataLog(EVENTS, schema=EventSchema, id_range=IDS,
                      dt_range=EVENTS_TME)
    assert log[2].data.label is None
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    EVENTS.loc[0, 'label'] = ''
    log = dtl.DataLog(EVENTS, schema=EventSchema, id_range=IDS,
                      dt_range=EVENTS_TME)
    assert log[2].data.label is None
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    r = log[2]
    seq = dtl.DataSequence.from_records([r],
                                        id_range=r.id,
                                        dt_range=(r.datetime,))
    assert seq[0] == r


def test_states():
    log = dtl.DataLog(STATES, schema=StateSchema, id_range=IDS,
                      dt_range=STATES_TME)
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    states = STATES.copy()
    states.loc[0, 'label'] = 'invalid_label'
    log = dtl.DataLog(states, schema=StateSchema, id_range=IDS,
                      dt_range=STATES_TME)
    assert log['88:4A:EA:69:35:BD'][-1].label is None
    assert log['88:4A:EA:69:35:BD']['2018-4-15 00:16:10'].label == 'Loading'
    t0 = '2018-4-15 00:16:10'
    t1 = '2018-4-15 00:16:30'
    assert len(log['88:4A:EA:69:35:BD'][t0:t1]) == 5
    logslice = log[t0:t1]
    assert isinstance(logslice, dtl.DataLog)
    assert len(logslice) == 7
    logarray = log[slice(None), t0]
    assert isinstance(logarray, dtl.DataArray)
    assert len(logarray) == 2
    assert log['2018-4-13':'2018-4-14'].empty
    assert len(log['2018-4-15 00:16:59.999':'2018-4-17']) == 2
    assert log['2018-4-16':'2018-4-17'].empty


def test_sessions():
    log = dtl.DataLog(SESSIONS, schema=SessionSchema, id_range=IDS,
                      dt_range=SESSIONS_TME)
    assert dtl.DataLog.json_loads(log.json_dumps()) == log


def test_periods():
    log = dtl.DataLog(PERIODS, schema=PeriodSchema, id_range=IDS,
                      dt_range=PERIODS_TME)
    assert dtl.DataLog.json_loads(log.json_dumps()) == log
    PERIODS['quota'] = PERIODS.quota.apply(lambda x: True if x else 'false')
    log2 = dtl.DataLog(PERIODS, schema=PeriodSchema, id_range=IDS,
                       dt_range=PERIODS_TME)
    assert log == log2
    PERIODS['oee'][0] = np.nan
    PERIODS['oee'][1] = 101
    log3 = dtl.DataLog(PERIODS, schema=PeriodSchema, id_range=IDS,
                       dt_range=PERIODS_TME)
    assert log3[0].validate()
    assert not log3[1].validate()


def plots():
    samples = dtl.DataLog(SAMPLES, schema=SampleSchema, id_range=IDS,
                          dt_range=SAMPLES_TME)
    samples['88:4A:EA:69:35:BD'].plot()
    score = samples['88:4A:EA:69:35:BD'].plot(columns=['accel_energy_512', ])
    score.staves[0].plot_sample_sequence(samples['88:4A:EA:69:35:BD'],
                                         'temperature',
                                         twinx=True)
    empty_sequence = samples['88:4A:EA:69:35:BD']['2018-4-15 00:15:00':
                                                  '2018-4-15 00:15:00.250']
    empty_sequence.plot()

    events = dtl.DataLog(EVENTS, schema=EventSchema, id_range=IDS,
                         dt_range=EVENTS_TME)['88:4A:EA:69:35:BD']
    events._data.loc[events.index[0], 'label'] = None
    events.plot()

    sessions = dtl.DataLog(SESSIONS, schema=SessionSchema, id_range=IDS,
                           dt_range=SESSIONS_TME)
    sessions['88:4A:EA:69:35:BD'].plot(tz='Asia/Kolkata', ymin=0.25, ymax=0.75)

    states = dtl.DataLog(STATES, schema=StateSchema, id_range=IDS,
                         dt_range=STATES_TME)
    score = states['88:4A:EA:69:35:BD'].plot(ymin=0.5, ymax=1.)
    states['88:4A:EA:69:38:1A'].plot(score.staves[0],
                                     color={'Loading': 'grey',
                                            'Executing': 'green'},
                                     ymin=0., ymax=0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
    plots()
