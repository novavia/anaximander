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

import anaximander3 as nx
from anaximander3.utilities import nxtime
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl, records as rec
from anaximander3.io import gcbigtable as gbt

# PROJECT_ID = 'anaximander-tests'
# INSTANCE_ID = 'testinstance'
PROJECT_ID = 'infinite-uptime-test'
INSTANCE_ID = 'testinstance'
INSTANCE_LOC = 'us-central1-c'

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


class BtSchema(sch.SampleLogsSchema):
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


@pytest.fixture(scope="module")
def featurelog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id',
                              'Feature_Value_1': 'accel_x',
                              'Feature_Value_2': 'accel_y'},
                     inplace=True)
    log = dtl.DataLog(dataframe, schema=BtSchema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    return log


def cleanup(instance):
    """Cleans up the supplied Bigtable instance."""
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
    return gbt.BigTableDataTable(instance, 'ghost', BtSchema)


@pytest.fixture(scope="module")
def empty_table(instance):
    """Yields an empty table to test insertions and appends."""
    table = gbt.BigTableDataTable(instance, 'empty', BtSchema)
    table.create(warn=False, overwrite=True, force=True)
    return table


@pytest.fixture(scope="module")
def full_table(instance, featurelog):
    """Yields a populated table to test queries."""
    table = gbt.BigTableDataTable(instance, 'full', BtSchema)
    table.create(warn=False, overwrite=True, force=True)
    # Populates the table with some data
    table.append(featurelog)
    return table


@pytest.fixture(scope="module")
def refuse_table(instance, featurelog):
    """Yields a populated table to test deletes."""
    table = gbt.BigTableDataTable(instance, 'refuse', BtSchema)
    table.create(warn=False, overwrite=True, force=True)
    # Populates the table with some data
    table.append(featurelog)
    return table

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud, pytest.mark.bigtable]


def test_table_instantiation(ghost_table):
    assert ghost_table.table_id == 'ghost'
    assert ghost_table.reverse


def test_keymaker(featurelog, ghost_table):
    record = featurelog[0]
    key = '3C:ED:70:91:E9:86#2628597572200000'
    assert ghost_table.schema.rowkey(record.idx) == key


def test_record(full_table):
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:02:27.800000+00:00'))
    record = full_table.record(idx)
    assert record.id == '88:4A:EA:69:DF:A2'
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:02:27.900000+00:00'))
    with pytest.raises(KeyError):
        full_table.record(*idx)


def test_insert(empty_table, featurelog):
    record = featurelog[0]
    empty_table.insert(record)
    row = empty_table.table.read_row(record.key.encode('utf-8'))
    assert isinstance(row, bt.row_data.PartialRowData)


def test_update(empty_table, featurelog):
    record = featurelog[-1]
    record_x = record('accel_x')
    empty_table.insert(record_x)
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:04:58.700000+00:00'))
    assert empty_table.record(idx) == record_x
    record_y = record('accel_y')
    empty_table.update(record_y)
    assert empty_table.record(idx) == record


def test_append(empty_table, featurelog):
    empty_table.append(featurelog)
    rowdata = empty_table.table.read_rows()
    rowdata.consume_all()
    assert len(rowdata.rows) >= len(featurelog)


def test_query(full_table, featurelog):
    query = full_table.query(id=DEVICES)
    assert query.schema == BtSchema()
    assert len(list(query.fetch())) == len(featurelog)
    assert len(list(query.fetch(limit=1))) == 2


def test_first(full_table):
    query = full_table.query(id='88:4A:EA:69:DF:A2')
    record = query.first()
    assert isinstance(record, rec.Record)
    assert record.id == '88:4A:EA:69:DF:A2'
    assert record.datetime == pd.Timestamp('2016-09-14 10:04:58.700000+00:00')
    query = full_table.query(id='88:4A:EA:69:DF:A2',
                             datetime=(None, '2016-09-14 10:03:00'))
    record = query.first()
    assert record.datetime == pd.Timestamp('2016-09-14 10:02:27.800000+00:00')


def test_log(full_table):
    query = full_table.query(id='68:9E:19:07:DE:C3')
    log = query.data()
    assert isinstance(log, dtl.DataSequence)
    assert log.id_range == '68:9E:19:07:DE:C3'
    assert len(log) == 3
    query = full_table.query(id=DEVICES)


def test_fields(full_table):
    query = full_table.query('accel_x', id='68:9E:19:07:DE:C3')
    log = query.data()
    assert log.schema == BtSchema('accel_x')


def test_delete(refuse_table):
    idx = ('68:9E:19:07:DE:C3',
           pd.Timestamp('2016-09-14 10:00:27.800000+00:00'))
    refuse_table.delete(idx)
    query = refuse_table.query(id='68:9E:19:07:DE:C3')
    assert len(query.data()) == 2


def test_discard(refuse_table):
    start = '2016-9-14 10:01'
    end = '2016-9-14 10:05'
    refuse_table.discard(id='88:4A:EA:69:DF:A2', datetime=(start, end))
    query = refuse_table.query(id='88:4A:EA:69:DF:A2')
    assert len(query.data()) == 5
    refuse_table.discard(id='68:9E:19:07:DE:C3')
    query = refuse_table.query(id='68:9E:19:07:DE:C3')
    with pytest.raises(gbt.EmptyQueryException):
        query.first()


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb', '--bigtable'])
