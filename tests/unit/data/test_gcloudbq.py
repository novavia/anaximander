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

import pytest

from gcloud import bigquery as bq

from anaximander.data import data, fields as fld, schema as sch, \
    tract as trc, gcloudbq as gbq


PROJECT_ID = 'infinite-uptime-1232'
BQ = bq.Client(PROJECT_ID)
DATASET_ID = 'InterfaceTesting'

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Test Cases
# =============================================================================


def cleanup(dataset):
    """Cleans up the supplied BigQuery dataset."""
    for table in dataset.list_tables()[0]:
        table.delete()


@pytest.fixture
def dataset():
    dataset_ = BQ.dataset(DATASET_ID)
    if dataset_.exists():
        cleanup(dataset_)
    else:
        dataset_.create()
    yield dataset_  # provide the fixture value
    cleanup(dataset_)


test_domain = trc.DataDomain('test')


@trc.domain('test')
@trc.tract
class DeviceData(sch.Schema):
    mac = fld.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fld.DateTime(key=True, sequential=True)
    accel_x = fld.Scalar(data.NxFloat)


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


def test_create_table(dataset):
    table = gbq.create_table(dataset, DeviceData)
    assert table.exists()
    assert table.friendly_name == 'DeviceData'


def test_create_all(dataset):
    tables = gbq.create_all(dataset, test_domain)
    table = list(tables)[0]
    assert table.exists()
    assert table.friendly_name == 'DeviceData'


if __name__ == '__main__':
    pytest.main([__file__])
