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
from anaximander3.utilities import nxtime
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

FEATURE_IDS = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
FEATURE_TME = ['2016-9-14 10:00', '2016-9-14 10:05']
MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
DEVICES = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']

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

GHOST = Title('ghost', RdSchema)
EMPTY = Title('empty', RdSchema)
MAX_COUNT = Title('max_count', RdSchema)
MAX_RANGE = Title('max_range', RdSchema)
MAX_MAX = Title('max_max', RdSchema)
FULL = Title('full', RdSchema)
REFUSE = Title('refuse', RdSchema)


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
    return nxr.RedisTract(store, GHOST)


@pytest.fixture(scope="module")
def empty_tract(store):
    """Yields an empty tract to test insertions and appends."""
    tract = nxr.RedisTract(store, EMPTY)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def max_count_tract(store):
    """Yields an empty tract with a max_count."""
    tract = nxr.RedisTract(store, MAX_COUNT, max_count=5)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def max_range_tract(store):
    """Yields an empty tract with a max_range."""
    tract = nxr.RedisTract(store, MAX_RANGE, max_range=1)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def max_max_tract(store):
    """Yields an empty tract with a max_count and max_range."""
    tract = nxr.RedisTract(store, MAX_MAX, max_range=1, max_count=3)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def full_tract(store, featurelog):
    """Yields a populated tract to test queries."""
    tract = nxr.RedisTract(store, FULL)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def refuse_tract(store, featurelog):
    """Yields a populated tract to test deletes."""
    tract = nxr.RedisTract(store, REFUSE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud]


def test_tract_instantiation(ghost_tract):
    assert ghost_tract.name == 'ghost'
    assert ghost_tract.reverse
    assert ghost_tract.max_range is None
    assert ghost_tract.max_count is None


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


def test_max_count(max_count_tract, featurelog):
    client = max_count_tract.client
    max_count_tract.append(featurelog)
    assert client.zcard('max_count#68:9E:19:07:DE:C3') == 3
    assert client.zcard('max_count#88:4A:EA:69:DF:A2') == 5
    query = max_count_tract.query(id='88:4A:EA:69:DF:A2')
    ts = pd.Timestamp('2016-09-14 10:03:56.670000+00:00')
    assert query.data()[0].datetime == ts
    client.zremrangebyrank('max_count#88:4A:EA:69:DF:A2', 0, 0)
    assert client.zcard('max_count#88:4A:EA:69:DF:A2') == 4
    record = featurelog[-10]
    max_count_tract.insert(record)
    data = query.data()
    assert len(data) == 5
    assert data[0] == record
    record = featurelog[-9]
    max_count_tract.insert(record)
    data = query.data()
    assert len(data) == 5
    assert data[0] == record


def test_max_range(max_range_tract, featurelog):
    client = max_range_tract.client
    max_range_tract.append(featurelog)
    assert client.zcard('max_range#68:9E:19:07:DE:C3') == 1
    assert client.zcard('max_range#88:4A:EA:69:DF:A2') == 4
    query = max_range_tract.query(id='88:4A:EA:69:DF:A2')
    ts = pd.Timestamp('2016-09-14 10:04:00.690000+00:00')
    assert query.data()[0].datetime == ts
    record = featurelog[4]
    max_range_tract.insert(record)
    data = query.data()
    assert len(data) == 5
    assert data[0] == record
    record = featurelog[-1]
    max_range_tract.insert(record)
    data = query.data()
    assert len(data) == 4
    assert data[-1] == record


def test_max_max(max_max_tract, featurelog):
    client = max_max_tract.client
    max_max_tract.append(featurelog)
    assert client.zcard('max_max#68:9E:19:07:DE:C3') == 1
    assert client.zcard('max_max#88:4A:EA:69:DF:A2') == 3
    query = max_max_tract.query(id='88:4A:EA:69:DF:A2')
    record = featurelog[4]
    max_max_tract.insert(record)
    data = query.data()
    assert len(data) == 3
    assert data[0] != record
    record = featurelog[-1]
    max_max_tract.delete(record.idx)
    data = query.data()
    assert len(data) == 2
    record = featurelog[4]
    max_max_tract.insert(record)
    data = query.data()
    assert len(data) == 3
    assert data[0] == record
    record = featurelog[-1]
    max_max_tract.insert(record)
    data = query.data()
    assert len(data) == 3
    assert data.datetime[-1] - data.datetime[0] <= max_max_tract.max_range


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


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
