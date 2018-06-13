#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for BigQuery interface.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import os.path

import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.utilities import nxtime, nxrange as rge
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl, records as rec
from anaximander3.io.store import Title, EmptyQueryException
from anaximander3.io import redis as nxr


HOST = 'redis-15511.c1.us-central1-2.gce.cloud.redislabs.com'
PORT = 15511
PWD = '73wDWoBe'

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')
STATES_PATH = os.path.join(TEST_DATA_DIR, 'states.csv')
STATES = pd.read_csv(STATES_PATH)

FEATURE_IDS = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
FEATURE_TME = ['2016-9-14 10:00', '2016-9-14 10:05']
MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
DEVICES = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
IDS = ['88:4A:EA:69:35:BD', '88:4A:EA:69:38:1A']
STATES_TME = ['2018-4-15 00:16:00', '2018-4-15 00:17:00']

# =============================================================================
# Environment
# =============================================================================


class RdSchema(sch.SampleLogsSchema):
    accel_x = cln.Measurement()
    accel_y = cln.Measurement()

    def __rowkey__(self, index):
        id, dt = index
        postfix = str(int(1e6 * (nxtime.MAX_TIMESTAMP - dt.timestamp())))
        return '#'.join((id[::-1], postfix))

    def __rowidx__(self, key):
        id, postfix = key.split('#')
        dt = pd.Timestamp(1e6 * nxtime.MAX_TIMESTAMP - int(postfix),
                          tz='UTC', unit='us')
        return (id[::-1], dt)


class StateSchema(sch.StateLogsSchema):
    label = cln.StateLabel(('Loading', 'Executing'))


GHOST = Title('ghost', RdSchema)
EMPTY = Title('empty', RdSchema)
FULL = Title('full', RdSchema)
REFUSE = Title('refuse', RdSchema)
STATE = Title('state', StateSchema)


@pytest.fixture(scope="module")
def featurelog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id',
                              'Feature_Value_1': 'accel_x',
                              'Feature_Value_2': 'accel_y'},
                     inplace=True)
    log = dtl.DataLog(dataframe, schema=RdSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    return log


@pytest.fixture(scope="module")
def statelog():
    """Returns a nominal dataframe."""
    log = dtl.DataLog(STATES, schema=StateSchema, id_range=IDS,
                      dt_range=STATES_TME)
    return log


def cleanup(store):
    """Cleans up the Redis instance."""
    store.io.flushdb()
    store.io.connection_pool.disconnect()


@pytest.fixture(scope="session")
def store():
    """Creates a redis store for testing purposes."""
    store = nxr.RedisStore(host=HOST, port=PORT, password=PWD)
    yield store
    cleanup(store)


@pytest.fixture(scope="module")
def ghost_tract(store):
    """Yields uncreated tract store."""
    return nxr.RedisDataTract(store, GHOST)


@pytest.fixture(scope="module")
def empty_tract(store):
    """Yields an empty tract to test insertions and appends."""
    tract = nxr.RedisDataTract(store, EMPTY)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def full_tract(store, featurelog):
    """Yields a populated tract to test queries."""
    tract = nxr.RedisDataTract(store, FULL)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def refuse_tract(store, featurelog):
    """Yields a populated tract to test deletes."""
    tract = nxr.RedisDataTract(store, REFUSE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def state_tract(store, statelog):
    """Yields a populated tract to test xindex queries."""
    tract = nxr.RedisDataTract(store, STATE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(statelog)
    return tract


# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud]


def test_tract_instantiation(ghost_tract):
    assert ghost_tract.name == 'ghost'


def test_keymaker(featurelog, ghost_tract):
    record = featurelog[0]
    key = '3C:ED:70:91:E9:86#2628597572200000'
    assert ghost_tract.schema.rowkey(record.idx) == key


def test_record(full_tract):
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:02:27.800000+00:00'))
    record = full_tract.record(idx)
    assert record.id == '88:4A:EA:69:DF:A2'
    keys = ('88:4A:EA:69:DF:A2',
            pd.Timestamp('2016-09-14 10:02:27.900000+00:00'))
    with pytest.raises(KeyError):
        full_tract.record(*keys)
    records = full_tract.records(idx, keys)
    assert next(records).id == '88:4A:EA:69:DF:A2'
    with pytest.raises(KeyError):
        next(records)


def test_insert(empty_tract, featurelog):
    record = featurelog[0]
    empty_tract.insert(record)
    assert not empty_tract.empty


def test_update(empty_tract, featurelog):
    record = featurelog[-1]
    record_x = record('accel_x')
    empty_tract.insert(record_x)
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:04:58.700000+00:00'))
    assert empty_tract.record(idx) == record_x
    record_y = record('accel_y')
    empty_tract.update(record_y)
    assert empty_tract.record(idx) == record


def test_append(empty_tract, featurelog):
    empty_tract.append(featurelog)
    assert empty_tract.client.zcard('empty#68:9E:19:07:DE:C3') >= 3


def test_query(full_tract, featurelog):
    query = full_tract.query(id=DEVICES)
    assert query.schema == RdSchema()
    assert len(list(query.fetch())) == len(featurelog)
    assert len(list(query.fetch(limit=1))) == 2


def test_first(full_tract):
    query = full_tract.query(id='88:4A:EA:69:DF:A2')
    record = query.first()
    assert isinstance(record, rec.Record)
    assert record.id == '88:4A:EA:69:DF:A2'
    assert record.datetime == pd.Timestamp('2016-09-14 10:04:58.700000+00:00')
    query = full_tract.query(id='88:4A:EA:69:DF:A2',
                             datetime=(None, '2016-09-14 10:03:00'))
    record = query.first()
    assert record.datetime == pd.Timestamp('2016-09-14 10:02:27.800000+00:00')


def test_data(full_tract):
    query = full_tract.query(id='68:9E:19:07:DE:C3')
    log = query.data()
    assert isinstance(log, dtl.DataSequence)
    assert log.id_range == '68:9E:19:07:DE:C3'
    assert len(log) == 3
    query = full_tract.query(id=DEVICES)


def test_fields(full_tract):
    query = full_tract.query('accel_x', id='68:9E:19:07:DE:C3')
    log = query.data()
    assert log.schema == RdSchema('accel_x')


def test_delete(refuse_tract):
    idx = ('68:9E:19:07:DE:C3',
           pd.Timestamp('2016-09-14 10:00:27.800000+00:00'))
    refuse_tract.delete(idx)
    query = refuse_tract.query(id='68:9E:19:07:DE:C3')
    assert len(query.data()) == 2


def test_discard(refuse_tract):
    start = '2016-9-14 10:01'
    end = '2016-9-14 10:05'
    refuse_tract.discard('88:4A:EA:69:DF:A2', datetime=(start, end))
    query = refuse_tract.query(id='88:4A:EA:69:DF:A2')
    assert len(query.data()) == 5
    refuse_tract.discard('68:9E:19:07:DE:C3')
    query = refuse_tract.query(id='68:9E:19:07:DE:C3')
    with pytest.raises(EmptyQueryException):
        query.first()


def test_xindex(state_tract):
    start = '2018-4-15 00:15'
    end = '2018-4-15 00:16:30'
    query = state_tract.query(id=IDS, datetime=(start, end))
    log = query.data()
    assert len(log) == 8
    assert log.dt_range == rge.time_range(start, end)
    start = '2018-4-15 00:16:30'
    end = '2018-4-15 00:17:00'
    query = state_tract.query(id=IDS, datetime=(start, end))
    log = query.data()
    assert len(log) == 11
    assert log.dt_range == rge.time_range(start, end)
    start = '2018-4-15 00:17:00'
    end = '2018-4-15 00:17:30'
    query = state_tract.query(id=IDS, datetime=(start, end))
    log = query.data()
    assert len(log) == 2
    assert log.dt_range == rge.time_range(start, end)

if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
