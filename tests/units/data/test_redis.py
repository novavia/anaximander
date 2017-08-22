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
import time

import pandas as pd
import pytest

import anaximander as nx
from anaximander import data as dat
from anaximander.data import fields as fld, schema as sch, redis as nxr


HOST = 'redis-15362.c1.us-central1-2.gce.cloud.redislabs.com'
PORT = 15362
PWD = '73wDWoBe'
CONNECTIONS = 30

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
DEVICES = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']

# =============================================================================
# Environment
# =============================================================================


@dat.tract
class DumbDeviceData(sch.Schema):
    mac = fld.ReStr(key='reverse', pattern=MAC_PATTERN)
    timestamp = fld.Timestamp(key='timestamp', sequential=True)
    accel_x = fld.Scalar(dat.NxFloat, family='features')


@dat.tract
class DeviceData(sch.TimeSchema):
    mac = fld.ReStr(key='reverse', pattern=MAC_PATTERN)
    timestamp = fld.Timestamp(key='timestamp', sequential=True)
    accel_x = fld.Scalar(dat.NxFloat, family='features')


@pytest.fixture(scope="module")
def frame():
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    dataframe['mac'] = dataframe.device
    dataframe['accel_x'] = dataframe.Feature_Value_1
    return DeviceData.Frame(dataframe)


def cleanup(client):
    """Cleans up the supplied redis client."""
    client.flushdb()
    client.connection_pool.disconnect()


@pytest.fixture(scope="session")
def instance():
    """Creates a redis instance for testing purposes."""
    client = nxr.client(HOST, PORT, PWD, CONNECTIONS)
    yield client
    cleanup(client)


@pytest.fixture(scope="module")
def ghost_table(instance):
    """Yields uncreated table instance."""
    return nxr.RedisDataTable[DumbDeviceData.Schema](instance, 'ghost',
                                                     retention=1)


@pytest.fixture(scope="module")
def empty_table(instance):
    """Yields an empty table to test insertions and appends."""
    table = nxr.RedisDataTable[DeviceData.Schema](instance, 'empty')
    table.create(warn=False, remove=True)
    return table


@pytest.fixture(scope="module")
def full_table(instance, frame):
    """Yields a populated table to test queries."""
    table = nxr.RedisDataTable[DeviceData.Schema](instance, 'full',
                                                  maxrate=150)
    table.create(warn=False, remove=True)
    # Populates the table with some data
    table.append(frame)
    return table

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud]


def test_table_instantiation(ghost_table):
    assert ghost_table.name == 'ghost'


def test_keymaker(frame):
    scoring = nxr.scoremaker(DeviceData.Schema)
    record = frame.iloc[0]
    ts = record.timestamp.timestamp()
    assert scoring(record.timestamp) == 1e6 * (4102444800 - ts)


def test_insert(empty_table, frame):
    record = frame.iloc[0]
    empty_table.insert(record)
    q = empty_table.query(mac=record.mac)
    assert q.first().timestamp == record.timestamp


def test_query(full_table, frame):
    query = full_table.query(mac=DEVICES)
    assert len(list(query.fetch())) == len(frame)
    assert len(list(query.fetch(maxrows=5))) == 5


def test_exclusion(full_table, frame):
    query = full_table.query(mac='68:9E:19:07:DE:C3',
                             timestamp=('2016-9-14 10:00', '2016-9-14 10:01'))
    assert len(list(query.fetch())) == 2
    query = full_table.query(mac='68:9E:19:07:DE:C3',
                             timestamp=('2016-9-14 10:00',
                                        '2016-9-14 10:00:29.5'))
    assert len(list(query.fetch())) == 1


def test_retention(ghost_table, frame):
    record = frame.iloc[0]
    ghost_table.insert(record)
    time.sleep(2)
    q = ghost_table.query(mac=record.mac)
    assert q.data().empty


def test_first(full_table):
    query = full_table.query(mac='88:4A:EA:69:DF:A2')
    record = query.first()
    assert type(record) == DeviceData.Record
    assert record.mac == '88:4A:EA:69:DF:A2'
    assert record.timestamp == pd.Timestamp('2016-09-14 10:04:58.700000+00:00')
    query = full_table.query(mac='88:4A:EA:69:DF:A2',
                             timestamp=(None, '2016-09-14 10:03:00'))
    record = query.first()
    assert record.timestamp == pd.Timestamp('2016-09-14 10:02:27.800000+00:00')


def test_maxrows(ghost_table, full_table):
    query = full_table.query(mac=DEVICES)
    assert query._maxrows == 1e5
    with pytest.raises(nxr.DataQueryException):
        query.frame(maxrows=5, maxraise=True)
    query = full_table.query(mac='88:4A:EA:69:DF:A2',
                             timestamp=('2016-09-14 10:00:00',
                                        '2016-09-14 10:03:00'))
    assert query._maxrows == 450
    query = ghost_table.query(mac='88:4A:EA:69:DF:A2',
                              timestamp=('2016-09-14 10:00:00',
                                         '2016-09-14 10:03:00'))
    assert query._maxrows == 1e5


def test_frame(full_table):
    query = full_table.query(mac='68:9E:19:07:DE:C3')
    frame = query.frame()
    assert type(frame) == DeviceData.Frame
    assert frame.mac.unique == {'68:9E:19:07:DE:C3'}
    assert len(frame) == 3
    query = full_table.query(mac=DEVICES)
    frame = query.frame(maxrows=1)
    assert len(frame) == 1


def test_fields(full_table):
    query = full_table.query('mac', 'timestamp', mac='68:9E:19:07:DE:C3')
    frame = query.frame()
    assert set(frame.data.columns) == {'mac', 'timestamp'}


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb', '--bigtable'])
