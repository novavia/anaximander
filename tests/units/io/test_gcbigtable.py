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
from anaximander3.utilities import nxtime, nxrange as rge
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl, records as rec
from anaximander3.io.store import Title
from anaximander3.io import gcbigtable as gbt

# PROJECT_ID = 'anaximander-tests'
# INSTANCE_ID = 'testinstance'
PROJECT_ID = 'infinite-uptime-test'
INSTANCE_ID = 'testinstance'
INSTANCE_LOC = 'us-central1-c'

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')
STATES_PATH = os.path.join(TEST_DATA_DIR, 'states.csv')
STATES = pd.read_csv(STATES_PATH)
SESSIONS_PATH = os.path.join(TEST_DATA_DIR, 'sessions.csv')
SESSIONS = pd.read_csv(SESSIONS_PATH)
MULTIEVENTS_PATH = os.path.join(TEST_DATA_DIR, 'multievents.csv')
MULTISESSIONS_PATH = os.path.join(TEST_DATA_DIR, 'multisessions.csv')

FEATURE_IDS = ['68:9E:19:07:DE:C3', '88:4A:EA:69:DF:A2']
FEATURE_TME = ['2016-9-14 10:00', '2016-9-14 10:05']
MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
DEVICES = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
IDS = ['88:4A:EA:69:35:BD', '88:4A:EA:69:38:1A']
STATES_TME = ['2018-4-15 00:16:00', '2018-4-15 00:17:00']
EVENTS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
SESSIONS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
MULTIEVENTS = pd.read_csv(MULTIEVENTS_PATH)
MULTISESSIONS = pd.read_csv(MULTISESSIONS_PATH)

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


class StateSchema(sch.StateLogsSchema):
    label = cln.StateLabel(('Loading', 'Executing'))


class SessionSchema(sch.SessionLogsSchema):
    part_id = cln.Integer()


class MultiEventSchema(sch.MultiEventLogsSchema):
    message = cln.Text()
    latency = cln.Float()


class MultiSessionSchema(sch.MultiSessionLogsSchema):
    pass


class CompoundStateSchema(sch.CompoundStateLogsSchema):
    pass


class GnSchema(sch.SampleLogsSchema):
    feature = cln.Measurement(generic=True)

    def __rowkey__(self, index):
        id, dt = index
        postfix = str(int(1e6 * (nxtime.MAX_TIMESTAMP - dt.timestamp())))
        return '#'.join((id[::-1], postfix))

    def __rowidx__(self, key):
        id, postfix = key.split('#')
        dt = pd.Timestamp(1e6 * nxtime.MAX_TIMESTAMP - int(postfix),
                          tz='UTC', unit='us')
        return (id[::-1], dt)


GHOST = Title('ghost', BtSchema)
EMPTY = Title('empty', BtSchema)
FULL = Title('full', BtSchema)
REFUSE = Title('refuse', BtSchema)
STATE = Title('state', StateSchema)
SESSION = Title('session', SessionSchema)
EVENT = Title('event', MultiEventSchema)
CSTATE = Title('cstate', CompoundStateSchema)
GENERIC = Title('generic', GnSchema)


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


@pytest.fixture(scope="module")
def statelog():
    """Returns a nominal dataframe."""
    log = dtl.DataLog(STATES, schema=StateSchema, id_range=IDS,
                      dt_range=STATES_TME)
    return log


@pytest.fixture(scope="module")
def sessionlog():
    """Returns a nominal dataframe."""
    log = dtl.DataLog(SESSIONS, schema=SessionSchema, id_range=IDS,
                      dt_range=SESSIONS_TME)
    return log


@pytest.fixture(scope="module")
def eventseq():
    """Returns a nominal dataframe."""
    seq = dtl.DataSequence(MULTIEVENTS, schema=MultiEventSchema,
                           id_range='88:4A:EA:69:35:BD', dt_range=EVENTS_TME)
    return seq


@pytest.fixture(scope="module")
def cstateseq():
    """Returns a nominal dataframe."""
    seq = dtl.DataSequence(MULTISESSIONS, schema=MultiSessionSchema,
                           id_range='88:4A:EA:69:35:BD',
                           dt_range=SESSIONS_TME).to_compound_state_sequence()
    return seq


@pytest.fixture(scope="module")
def genlog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id',
                              'Feature_Value_1': 'accel_x',
                              'Feature_Value_2': 'accel_y'},
                     inplace=True)
    schema = GnSchema('feature#accel_x', 'feature#accel_y')
    log = dtl.DataLog(dataframe, schema=schema,
                      id_range=FEATURE_IDS, dt_range=FEATURE_TME)
    return log


def cleanup(store):
    """Cleans up the supplied Bigtable instance."""
    for table in store.io.list_tables():
        table.delete()
    store.io.delete()


@pytest.fixture(scope="session")
def store():
    """Creates a bigtable instance for testing purposes."""
    store = gbt.BigTableStore(project_id=PROJECT_ID,
                              instance_id=INSTANCE_ID, admin=True)
    try:
        store.create(INSTANCE_LOC, serve_nodes=1)
    except _Rendezvous:
        pass
    except gbt.BigTableAdminException:
        cleanup(store)
        store.create(INSTANCE_LOC)
    time.sleep(10)
#    store = gbt.BigTableStore(project_id=PROJECT_ID,
#                              instance_id=INSTANCE_ID, admin=True)
    yield store
    cleanup(store)


@pytest.fixture(scope="module")
def ghost_tract(store):
    """Yields uncreated tract instance."""
    return gbt.BigTableTract(store, GHOST)


@pytest.fixture(scope="module")
def empty_tract(store):
    """Yields an empty tract to test insertions and appends."""
    tract = gbt.BigTableTract(store, EMPTY)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def full_tract(store, featurelog):
    """Yields a populated tract to test queries."""
    tract = gbt.BigTableTract(store, FULL)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def refuse_tract(store, featurelog):
    """Yields a populated tract to test deletes."""
    tract = gbt.BigTableTract(store, REFUSE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def state_tract(store, statelog):
    """Yields a populated tract to test xindex queries."""
    tract = gbt.BigTableTract(store, STATE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(statelog)
    return tract


@pytest.fixture(scope="module")
def session_tract(store, sessionlog):
    """Yields a populated tract to test session queries."""
    tract = gbt.BigTableTract(store, SESSION)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(sessionlog)
    return tract


@pytest.fixture(scope="module")
def event_tract(store, eventseq):
    """Yields a populated tract to test multi-event queries."""
    tract = gbt.BigTableTract(store, EVENT)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(eventseq)
    return tract


@pytest.fixture(scope="module")
def cstate_tract(store, cstateseq):
    """Yields a populated tract to test multi-event queries."""
    tract = gbt.BigTableTract(store, CSTATE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(cstateseq)
    return tract


@pytest.fixture(scope="module")
def gen_tract(store, genlog):
    """Yields a populated tract to test generic columns."""
    tract = gbt.BigTableTract(store, GENERIC)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(genlog)
    return tract

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud, pytest.mark.bigtable]


def test_tract_instantiation(ghost_tract):
    assert ghost_tract.table_id == 'ghost'


def test_keymaker(featurelog, ghost_tract):
    record = featurelog[0]
    key = '3C:ED:70:91:E9:86#2628597572200000'
    assert ghost_tract.schema.rowkey(record.idx) == key


def test_record(full_tract):
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:02:27.800000+00:00'))
    record = full_tract.record(idx)
    assert record.id == '88:4A:EA:69:DF:A2'
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:02:27.900000+00:00'))
    with pytest.raises(KeyError):
        full_tract.record(*idx)


def test_insert(empty_tract, featurelog):
    record = featurelog[0]
    empty_tract.insert(record)
    row = empty_tract.table.read_row(record.key.encode('utf-8'))
    assert isinstance(row, bt.row_data.PartialRowData)


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
    rowdata = empty_tract.table.read_rows()
    rowdata.consume_all()
    assert len(rowdata.rows) >= len(featurelog)


def test_query(full_tract, featurelog):
    query = full_tract.query(id=DEVICES)
    assert query.schema == BtSchema()
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


def test_log(full_tract):
    query = full_tract.query(id='68:9E:19:07:DE:C3')
    log = query.data()
    assert isinstance(log, dtl.DataSequence)
    assert log.id_range == '68:9E:19:07:DE:C3'
    assert len(log) == 3
    query = full_tract.query(id=DEVICES)


def test_fields(full_tract):
    query = full_tract.query('accel_x', id='68:9E:19:07:DE:C3')
    log = query.data()
    assert log.schema == BtSchema('accel_x')


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
    with pytest.raises(gbt.EmptyQueryException):
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
    start = '2018-04-15 00:16:32.610+00:00'
    end = '2018-4-15 00:17:00'
    query = state_tract.query(id='88:4A:EA:69:35:BD', datetime=(start, end))
    log = query.data()
    assert len(log) == 4


def test_session(session_tract):
    start = '2018-4-15 00:16'
    end = '2018-4-15 00:17'
    query = session_tract.query(id=IDS, datetime=(start, end))
    log = query.data()
    assert len(log) == 9
    start = '2018-4-15 00:16:10'
    end = '2018-4-15 00:17'
    query = session_tract.query(id=IDS, datetime=(start, end))
    log = query.data()
    assert len(log) == 8


def test_multi_event(event_tract):
    start = '2018-04-15 00:15:30'
    end = '2018-04-15 00:16:30'
    query = event_tract.query(id='88:4A:EA:69:35:BD', datetime=(start, end))
    seq = query.data()
    assert isinstance(seq, dtl.MultiEventSequence)
    assert len(seq) == 1


def test_generic(gen_tract):
    query = gen_tract.query('feature#accel_x', id='88:4A:EA:69:DF:A2')
    sequence = query.data()
    assert list(sequence.payload) == ['accel_x']


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb', '--bigtable'])
