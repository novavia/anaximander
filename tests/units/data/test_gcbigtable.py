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

import anaximander as nx
from anaximander import data as dat
from anaximander.data import fields as fld, schema as sch, gcbigtable as gbt
from anaximander.data.table import DataTableWarning

PROJECT_ID = 'anaximander-tests'
INSTANCE_ID = 'testinstance'

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
DEVICES = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']

# =============================================================================
# Environment
# =============================================================================


@dat.tract
class DeviceData(sch.Schema):
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


@pytest.fixture(scope="module")
def ghost_table():
    """Yields uncreated table instance that is not meant to be populated."""
    btclient = bt.Client(project=PROJECT_ID, admin=True)
    instance = btclient.instance(INSTANCE_ID)
    table = gbt.BigTableDataTable[DeviceData.Schema](instance, 'ghost')
    yield table  # provide the fixture value
    cleanup(instance)


@pytest.fixture(scope="module")
def empty_table():
    """Yields an empty table to test insertions and appends."""
    btclient = bt.Client(project=PROJECT_ID, admin=True)
    instance = btclient.instance(INSTANCE_ID)
    table = gbt.BigTableDataTable[DeviceData.Schema](instance, 'empty')
    table.create(warn=False, remove=True)
    yield table  # provide the fixture value
    cleanup(instance)


@pytest.fixture(scope="module")
def full_table(frame):
    """Yields a populated table to test queries."""
    btclient = bt.Client(project=PROJECT_ID, admin=True)
    instance = btclient.instance(INSTANCE_ID)
    table = gbt.BigTableDataTable[DeviceData.Schema](instance, 'full')
    table.create(warn=False, remove=True)
    # Populates the table with some data
    table.append(frame)
    yield table  # provide the fixture value
    cleanup(instance)

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud]


def test_btcolumns():
    columns = gbt.btcolumns(*DeviceData.Schema.fields.values())
    assert columns == {'keys': ['mac', 'timestamp'],
                       'features': ['accel_x']}


def test_table_instantiation(ghost_table):
    assert ghost_table.table_id == 'ghost'


def test_keymaker(frame):
    rowkey = gbt.keymaker(DeviceData.Schema)
    record = frame.iloc[0]
    assert rowkey(*record.keys) == '2A:FD:96:AE:A4:88#1473847201520000'


def test_insert(empty_table, frame):
    record = frame.iloc[0]
    empty_table.insert(record)
    rowkey = empty_table.rowkey(*record.keys)
    row = empty_table.table.read_row(rowkey.encode('utf-8'))
    assert isinstance(row, bt.row_data.PartialRowData)


def test_append(empty_table, frame):
    empty_table.append(frame)
    rowdata = empty_table.table.read_rows()
    rowdata.consume_all()
    assert len(rowdata.rows) >= len(frame)


def test_query(full_table, frame):
    query = full_table.query(mac=DEVICES)
    assert query.columns == full_table.columns
    assert len(list(query.fetch())) == len(frame)


def test_first(full_table):
    query = full_table.query(mac='88:4A:EA:69:DF:A2')
    record = query.first()
    assert type(record) == DeviceData.Record
    assert record.mac == '88:4A:EA:69:DF:A2'


def test_frame(full_table):
    query = full_table.query(mac='68:9E:19:07:DE:C3')
    frame = query.frame()
    assert type(frame) == DeviceData.Frame
    assert frame.mac.unique == {'68:9E:19:07:DE:C3'}
    assert len(frame) == 3


def test_fields(full_table):
    query = full_table.query('mac', 'timestamp', mac='68:9E:19:07:DE:C3')
    frame = query.frame()
    assert set(frame.data.columns) == {'mac', 'timestamp'}


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
