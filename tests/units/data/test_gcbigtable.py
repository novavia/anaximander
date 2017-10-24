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

from google.cloud import bigtable as bt
from grpc._channel import _Rendezvous
from google.cloud.happybase.connection import Connection

import anaximander as nx
from anaximander import data as dat
from anaximander.data import fields as fld, schema as sch, gcbigtable as gbt

#PROJECT_ID = 'anaximander-tests'
#INSTANCE_ID = 'testinstance'
PROJECT_ID = 'infinite-uptime-1232'
INSTANCE_ID = 'testinstance'
INSTANCE_LOC = 'us-central1-c'

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


def cleanup(instance):
    """Cleans up the supplied BigQuery dataset."""
    for table in instance.list_tables():
        table.delete()
    instance.delete()


@pytest.fixture(scope="session")
def instance():
    """Creates a bigtable instance for testing purposes."""
    btclient = bt.Client(project=PROJECT_ID, admin=True)
    instance = btclient.instance(INSTANCE_ID, INSTANCE_LOC)
    instance.display_name = INSTANCE_ID
    try:
        instance.create()
    except _Rendezvous:
        pass
    time.sleep(10)
    yield instance
    cleanup(instance)


@pytest.fixture(scope="module")
def ghost_table(instance):
    """Yields uncreated table instance."""
    return gbt.BigTableDataTable[DumbDeviceData.Schema](instance, 'ghost')


@pytest.fixture(scope="module")
def empty_table(instance):
    """Yields an empty table to test insertions and appends."""
    table = gbt.BigTableDataTable[DeviceData.Schema](instance, 'empty')
    table.create(warn=False, remove=True)
    return table


@pytest.fixture(scope="module")
def full_table(instance, frame):
    """Yields a populated table to test queries."""
    table = gbt.BigTableDataTable[DeviceData.Schema](instance, 'full',
                                                     maxrate=150)
    table.create(warn=False, remove=True)
    # Populates the table with some data
    table.append(frame)
    return table

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud, pytest.mark.bigtable]


def test_btcolumns():
    columns = gbt.btcolumns(*DeviceData.Schema.fields.values())
    assert columns == {'keys': ['mac', 'timestamp'],
                       'features': ['accel_x']}


def test_table_instantiation(ghost_table):
    assert ghost_table.table_id == 'ghost'
    assert ghost_table.reverse


def test_keymaker(frame):
    rowkey = gbt.keymaker(DeviceData.Schema)
    record = frame.iloc[0]
    assert rowkey(*record.keys) == '2A:FD:96:AE:A4:88#2628597598480000'


def test_record(full_table):
    keys = ('88:4A:EA:69:DF:A2',
            pd.Timestamp('2016-09-14 10:02:27.800000+00:00'))
    record = full_table.record(*keys)
    assert record.mac == '88:4A:EA:69:DF:A2'
    keys = ('88:4A:EA:69:DF:A2',
            pd.Timestamp('2016-09-14 10:02:27.900000+00:00'))
    with pytest.raises(gbt.EmptyQueryException):
        full_table.record(*keys)


def test_insert(empty_table, frame):
    record = frame.iloc[0]
    empty_table.insert(record)
    rowkey = empty_table.rowkey(*record.keys)
    row = empty_table.table.read_row(rowkey.encode('utf-8'))
    assert isinstance(row, bt.row_data.PartialRowData)


def test_connection(empty_table):
    pool = empty_table.pool
    with pool.connection() as connection:
        assert isinstance(connection, Connection)


def test_append(empty_table, frame):
    empty_table.append(frame)
    rowdata = empty_table.table.read_rows()
    rowdata.consume_all()
    assert len(rowdata.rows) >= len(frame)


def test_query(full_table, frame):
    query = full_table.query(mac=DEVICES)
    assert query.columns == full_table.columns
    assert len(list(query.fetch())) == len(frame)
    assert len(list(query.fetch(maxrows=5))) == 5


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
    with pytest.raises(gbt.DataQueryException):
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
