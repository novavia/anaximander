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

from google.cloud import bigquery as bq

import anaximander as nx
from anaximander import data as dat
from anaximander.data import fields as fld, schema as sch, gcloudbq as gbq

PROJECT_ID = 'anaximander-tests'
DATASET_ID = 'InterfaceTesting'

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Environment
# =============================================================================


@dat.tract
class DeviceData(sch.Schema):
    mac = fld.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fld.Timestamp(key=True, sequential=True)
    accel_x = fld.Scalar(dat.NxFloat)


def featurelog():
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    dataframe['mac'] = dataframe.device
    dataframe['accel_x'] = dataframe.Feature_Value_1
    return dataframe


def cleanup(dataset):
    """Cleans up the supplied BigQuery dataset."""
    for table in dataset.list_tables():
        table.delete()


@pytest.fixture(scope="module")
def storage():
    BQ = bq.Client(PROJECT_ID)
    dataset = BQ.dataset(DATASET_ID)
    if dataset.exists():
        cleanup(dataset)
    else:
        dataset.create()
    table = gbq.BigQueryDataTable[DeviceData.Schema](dataset, 'DeviceData')
    table.create()
    # Populates the table with some data
    data = DeviceData.Frame(featurelog())
    table.append(data)
    yield dataset, table  # provide the fixture value
    cleanup(dataset)

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud]


def test_bqfield():
    mac = gbq.bqfield(DeviceData.Schema.mac)
    timestamp = gbq.bqfield(DeviceData.Schema.timestamp)
    accel_x = gbq.bqfield(DeviceData.Schema.accel_x)
    assert isinstance(mac, bq.SchemaField)
    assert isinstance(timestamp, bq.SchemaField)
    assert isinstance(accel_x, bq.SchemaField)
    assert mac.field_type is 'STRING'
    assert timestamp.field_type is 'TIMESTAMP'
    assert accel_x.field_type is 'FLOAT'


def test_bqtable_instance(storage):
    dataset, table = storage
    bqtable = gbq.BigQueryDataTable[DeviceData.Schema](dataset, 'DeviceData')
    assert bqtable.table_id == table.table_id


def test_sql_generator(storage):
    dataset, _ = storage
    bqtable = gbq.BigQueryDataTable[DeviceData.Schema](dataset, 'DeviceData')
    query = gbq.BigQueryQuery(bqtable, mac='68:9E:19:07:DE:C3')
    assert query.sql == "SELECT mac, timestamp, accel_x FROM " + \
        "[anaximander-tests:InterfaceTesting.DeviceData] " + \
        "WHERE mac = '68:9E:19:07:DE:C3' ORDER BY timestamp"
    query = gbq.BigQueryQuery(bqtable, mac='68:9E:19:07:DE:C3',
                              timestamp=('2016-9-14 10:04',))
    assert query.sql == "SELECT mac, timestamp, accel_x FROM " + \
        "[anaximander-tests:InterfaceTesting.DeviceData] " + \
        "WHERE mac = '68:9E:19:07:DE:C3' AND " + \
        "timestamp >= '2016-09-14 10:04:00+00:00' ORDER BY timestamp" 


def test_append(storage):
    _, table = storage
    data = DeviceData.Frame(featurelog())
    response = table.append(data)
    assert response == []


def test_insert(storage):
    _, table = storage
    data = DeviceData.Frame(featurelog())
    records = data.to_records()
    response = table.insert(*records)
    assert response == []


def test_fetch(storage):
    dataset, _ = storage
    bqtable = gbq.BigQueryDataTable[DeviceData.Schema](dataset, 'DeviceData')
    query = gbq.BigQueryQuery(bqtable, mac='68:9E:19:07:DE:C3')
    record_generator = query.fetch()
    assert type(next(record_generator)) == DeviceData.Record


def test_frame(storage):
    dataset, _ = storage
    bqtable = gbq.BigQueryDataTable[DeviceData.Schema](dataset, 'DeviceData')
    query = gbq.BigQueryQuery(bqtable, mac='68:9E:19:07:DE:C3')
    frame = query.frame(limit=3)
    assert type(frame) == DeviceData.Frame
    assert frame.mac.unique == {'68:9E:19:07:DE:C3'}
    assert len(frame) == 3


def test_query(storage):
    _, table = storage
    query = table.query()
    frame = query.frame(limit=5)
    assert type(frame) is DeviceData.Frame
    assert len(frame) == 5


def test_raw_query(storage):
    _, table = storage
    sql = "SELECT * FROM [{p}:{d}.{t}] LIMIT 5"
    sql = sql.format(p=PROJECT_ID, d=DATASET_ID, t=DeviceData.tbname)
    query = table.query(sql=sql)
    frame = query.frame()
    assert type(frame) is DeviceData.Frame
    assert len(frame) == 5


if __name__ == '__main__':
    pytest.main([__file__])
