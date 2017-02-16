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

from gcloud import bigquery as bq

import anaximander as nx
from anaximander.data import data, fields as fld, schema as sch, \
    tract as trc, gcloudbq as gbq


PROJECT_ID = 'anaximander-tests'
DATASET_ID = 'InterfaceTesting'

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Environment
# =============================================================================


test_domain = trc.DataDomain('test')


@trc.domain('test')
@trc.tract
class DeviceData(sch.Schema):
    mac = fld.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fld.DateTime(key=True, sequential=True)
    accel_x = fld.Scalar(data.NxFloat)


def featurelog():
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    dataframe['mac'] = dataframe.device
    dataframe['accel_x'] = dataframe.Feature_Value_1
    return dataframe


def cleanup(dataset):
    """Cleans up the supplied BigQuery dataset."""
    for table in dataset.list_tables()[0]:
        table.delete()


@pytest.fixture(scope="module")
def storage():
    BQ = bq.Client(PROJECT_ID)
    dataset = BQ.dataset(DATASET_ID)
    if dataset.exists():
        cleanup(dataset)
    else:
        dataset.create()
    tables = gbq.create_all(dataset, test_domain)
    table = list(tables)[0]
    # Populates the table with some data
    data = DeviceData.Frame(featurelog())
    channel = gbq.BigQueryChannel(DeviceData, table)
    channel.append(data)
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


def test_create_all(storage):
    _, table = storage
    assert table.exists()
    assert table.friendly_name == 'DeviceData'


def test_append(storage):
    _, table = storage
    data = DeviceData.Frame(featurelog())
    channel = gbq.BigQueryChannel(DeviceData, table)
    response = channel.append(data)
    assert response == []


def test_insert(storage):
    _, table = storage
    data = DeviceData.Frame(featurelog())
    channel = gbq.BigQueryChannel(DeviceData, table)
    records = data.to_records()
    response = channel.insert(*records)
    assert response == []


def test_query(storage):
    _, table = storage
    channel = gbq.BigQueryChannel(DeviceData, table)
    query = channel.query(limit=10)
    frame = query()
    assert type(frame) is DeviceData.Frame
    assert len(frame) == 10


def test_raw_query(storage):
    _, table = storage
    channel = gbq.BigQueryChannel(DeviceData, table)
    sql = "SELECT * FROM [{p}:{d}.{t}] LIMIT 10"
    sql = sql.format(p=PROJECT_ID, d=DATASET_ID, t=DeviceData.tbname)
    query = channel.rawquery(sql)
    frame = query()
    assert type(frame) is DeviceData.Frame
    assert len(frame) == 10


if __name__ == '__main__':
    pytest.main([__file__])
