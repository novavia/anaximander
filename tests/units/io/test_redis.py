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
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.utilities import nxrange as rge, nxtime
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl, records as rec
from anaximander3.io.store import Store, Title, EmptyQueryException, \
    archive
from anaximander3.io import redis as nxr


HOST = 'redis-15511.c1.us-central1-2.gce.cloud.redislabs.com'
PORT = 15511
PWD = '73wDWoBe'

ALT_HOST = 'redis-11587.c1.us-central1-2.gce.cloud.redislabs.com'
ALT_PORT = 11587

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
DEVICES = ['68:9E:19:07:DE:C3', '88:4A:EA:69:DF:A2']
IDS = ['88:4A:EA:69:35:BD', '88:4A:EA:69:38:1A']
STATES_TME = ['2018-4-15 00:16:00', '2018-4-15 00:17:00']
EVENTS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
SESSIONS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
MULTIEVENTS = pd.read_csv(MULTIEVENTS_PATH)
MULTISESSIONS = pd.read_csv(MULTISESSIONS_PATH)

# =============================================================================
# Environment
# =============================================================================


@archive(name='scroll')
class RdSchema(sch.SampleLogsSchema):
    accel_x = cln.Measurement()
    accel_y = cln.Measurement()


class GnSchema(sch.SampleLogsSchema):
    feature = cln.Measurement(generic=True)


class StateSchema(sch.StateLogsSchema):
    label = cln.StateLabel(('Loading', 'Executing'))


class SessionSchema(sch.SessionLogsSchema):
    part_id = cln.Integer()


class MultiEventSchema(sch.MultiEventLogsSchema):
    message = cln.Text()
    latency = cln.Float()


class MultiSessionSchema(sch.MultiSessionLogsSchema):
    pass


GHOST = Title('ghost', RdSchema)
EMPTY = Title('empty', RdSchema)
FULL = Title('full', RdSchema)
REFUSE = Title('refuse', RdSchema)
STATE = Title('state', StateSchema)
SESSION = Title('session', SessionSchema)
EVENT = Title('event', MultiEventSchema)
CSTATE = Title('cstate', sch.CompoundStateLogsSchema)
SCROLL = Title('scroll', RdSchema)
STATE_SCROLL = Title('state_scroll', StateSchema, archive={})
GENERIC = Title('generic', GnSchema, stores=['archive'])


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
def mstateseq():
    """Returns a nominal dataframe."""
    seq = dtl.DataSequence(MULTISESSIONS, schema=MultiSessionSchema,
                           id_range='88:4A:EA:69:35:BD',
                           dt_range=SESSIONS_TME)
    return seq


@pytest.fixture(scope="module")
def cstateseq(mstateseq):
    """Returns a nominal dataframe."""
    return mstateseq.to_compound_state_sequence()


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
    """Cleans up the Redis instance."""
    store.io.flushdb()
    store.io.connection_pool.disconnect()


@pytest.fixture(scope="module")
def archive():
    """Creates a redis store for testing purposes."""
    store = nxr.RedisArchive(role='archive',
                             host=HOST, port=PORT, password=PWD)
#    store.tract(SCROLL)
#    store.tract(STATE_SCROLL)
#    store.tract(GENERIC)
    yield store
    cleanup(store)


@pytest.fixture(scope="module")
def procstore(archive):
    """Creates a redis pipeline store for testing purposes."""
    store = nxr.RedisPipeline(role='pipeline',
                              host=HOST, port=PORT, password=PWD,
                              archive=archive)
    yield store
    cleanup(store)


@pytest.fixture(scope="module")
def backupstore():
    """Creates a backup pipeline store for testing purposes."""
    store = nxr.RedisPipeline(role='backup',
                              host=ALT_HOST, port=ALT_PORT, password=PWD)
    yield store
    cleanup(store)


@pytest.fixture(scope="module")
def appstore(archive):
    """Creates a redis application store for testing purposes."""
    store = nxr.RedisBuffer(role='buffer',
                            host=ALT_HOST, port=ALT_PORT,
                            password=PWD, archive=archive)
    yield store
    cleanup(store)


@pytest.fixture(scope="module")
def ghost_tract(archive):
    """Yields uncreated tract store."""
    return archive.tract(GHOST)


@pytest.fixture(scope="module")
def empty_tract(archive):
    """Yields an empty tract to test insertions and appends."""
    tract = archive.tract(EMPTY)
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def full_tract(archive, featurelog):
    """Yields a populated tract to test queries."""
    tract = archive.tract(FULL)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def refuse_tract(archive, featurelog):
    """Yields a populated tract to test deletes."""
    tract = archive.tract(REFUSE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(featurelog)
    return tract


@pytest.fixture(scope="module")
def state_tract(archive, statelog):
    """Yields a populated tract to test xindex queries."""
    tract = archive.tract(STATE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(statelog)
    return tract


@pytest.fixture(scope="module")
def session_tract(archive, sessionlog):
    """Yields a populated tract to test session queries."""
    tract = archive.tract(SESSION)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(sessionlog)
    return tract


@pytest.fixture(scope="module")
def event_tract(archive, eventseq):
    """Yields a populated tract to test multi-event queries."""
    tract = archive.tract(EVENT)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(eventseq)
    return tract


@pytest.fixture(scope="module")
def cstate_tract(archive, cstateseq):
    """Yields a populated tract to test multi-event queries."""
    tract = archive.tract(CSTATE)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(cstateseq)
    return tract


@pytest.fixture(scope="module")
def gen_tract(archive, genlog):
    """Yields a populated tract to test generic columns."""
    tract = archive.tract(GENERIC)
    tract.create(warn=False, overwrite=True, force=True)
    # Populates the tract with some data
    tract.append(genlog)
    return tract


@pytest.fixture(scope="module")
def scroll_tract(procstore):
    """Yields an empty scroll."""
    tract = procstore.tract(SCROLL)
    tract.create(warn=False, overwrite=True, force=True)
    for id in FEATURE_IDS:
        tract.setup(id)
    return tract


@pytest.fixture(scope="module")
def state_scroll_tract(procstore):
    """Yields an empty scroll."""
    tract = procstore.tract(STATE_SCROLL)
    tract.create(warn=False, overwrite=True, force=True)
    for id in IDS:
        tract.setup(id)
    return tract


@pytest.fixture(scope="module")
def app_full_tract(appstore):
    """Yields an empty scroll."""
    tract = appstore.tract(FULL, '2m')
    tract.create(warn=False, overwrite=True, force=True)
    return tract


@pytest.fixture(scope="module")
def app_state_tract(appstore):
    """Yields an empty scroll."""
    tract = appstore.tract(STATE, '2m')
    tract.create(warn=False, overwrite=True, force=True)
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
    records = full_tract.records(idx, keys, keyerrors=False)
    assert list(records) == [record]


def test_insert(empty_tract, featurelog):
    record = featurelog[0]
    empty_tract.insert(record)
    assert not empty_tract.empty


def test_insert_null_state(state_tract, statelog):
    record = statelog[0].nulled_copy()
    state_tract.update(record)
    assert state_tract.record(record.idx) == record
    log = state_tract.query(id=record.id).data()
    assert log.data['label'][0] is np.nan


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
    query = full_tract.query(id=DEVICES, datetime=(None,
                                                   '2016-9-14 10:00:27.8'))
    assert len(query.data()) == 3


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


def test_nxpipe(archive, full_tract, featurelog):
    idx = ('88:4A:EA:69:DF:A2',
           pd.Timestamp('2016-09-14 10:02:27.800000+00:00'))
    pipe = nxr.NxPipe(archive)
    pipe.discard(full_tract, '88:4A:EA:69:DF:A2')
    pipe.append(full_tract, featurelog)
    pipe.record(full_tract, idx)
    record = pipe.execute()[-1]
    assert record.id == '88:4A:EA:69:DF:A2'
    keys = ('88:4A:EA:69:DF:A2',
            pd.Timestamp('2016-09-14 10:02:27.900000+00:00'))
    pipe = nxr.NxPipe(archive)
    pipe.record(full_tract, *keys)
    with pytest.raises(KeyError):
        pipe.execute()
    pipe = nxr.NxPipe(archive)
    pipe.records(full_tract, idx, keys, keyerrors=False)
    assert pipe.execute() == [record]
    pipe = nxr.NxPipe(archive)
    pipe.query(full_tract, id=DEVICES)
    pipe.query(full_tract, id='88:4A:EA:69:DF:A2', kind='first')
    log, record = pipe.execute()
    assert log.data.equals(featurelog.data)
    assert record == log[-1]


def test_query_data(full_tract):
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
    record = event_tract.record(('88:4A:EA:69:35:BD',
                                 '2018-04-15 00:15:27.100000+00:00',
                                 'heartbeat'))
    assert record.label == 'heartbeat'
    record = event_tract.record(('88:4A:EA:69:35:BD',
                                 '2018-04-15 00:15:27.100000+00:00',
                                 'alert'))
    assert record.label == 'alert'


def test_multi_session(mstateseq, cstate_tract):
    start = '2018-04-15 00:15:00'
    end = '2018-04-15 00:17:00'
    query = cstate_tract.query(id='88:4A:EA:69:35:BD', datetime=(start, end))
    seq = query.data().to_multisession_sequence(schema=MultiSessionSchema)
    assert seq == mstateseq
    start = '2018-04-15 00:16:00'
    query = cstate_tract.query(id='88:4A:EA:69:35:BD', datetime=(start, end))
    seq = query.data().to_multisession_sequence(schema=MultiSessionSchema)
    assert len(seq) == 7


def test_generic(gen_tract):
    query = gen_tract.query('feature#accel_x', id='88:4A:EA:69:DF:A2')
    sequence = query.data()
    assert list(sequence.payload) == ['accel_x']


def test_scroll_setup_teardown(scroll_tract, featurelog):
    assert scroll_tract.metadata('68:9E:19:07:DE:C3')['lower'] is pd.NaT
    scroll_tract.teardown('68:9E:19:07:DE:C3')
    with pytest.raises(nxr.NotInStoreException):
        scroll_tract.metadata('68:9E:19:07:DE:C3')


def test_write_scroll(scroll_tract, featurelog):
    with pytest.raises(TypeError):
        scroll_tract.write(featurelog)
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    scroll_tract.write(input_sequence)
    sequence = scroll_tract.sequence('68:9E:19:07:DE:C3')
    assert sequence.data.equals(input_sequence.data)
    assert sequence.dt_range == input_sequence.dt_range
    assert sequence.certification is pd.NaT
    assert sequence.consumption is nxtime.MAX


def test_metadata(scroll_tract, featurelog):
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    scroll_tract.write(input_sequence)
    metadata = scroll_tract.metadata('68:9E:19:07:DE:C3')
    assert set(metadata.keys()) == {'lower', 'upper', 'certification'}
    assert metadata['lower'] == input_sequence.dt_range.lower
    assert metadata['upper'] == input_sequence.dt_range.upper
    assert metadata['certification'] is pd.NaT
    subscriber = MagicMock()
    name_property = PropertyMock(return_value='subscriber')
    type(subscriber).name = name_property
    scroll_tract.update_subscriber(subscriber,
                                   '68:9E:19:07:DE:C3',
                                   '2016-9-14 10:01')
    metadata = scroll_tract.metadata('68:9E:19:07:DE:C3')
    assert set(metadata.keys()) == {'lower', 'upper', 'certification',
                                    'subscriber'}
    assert metadata['certification'] is pd.NaT
    assert metadata['subscriber'] == pd.to_datetime('2016-9-14 10:01',
                                                    utc=True)
    sequence = scroll_tract.sequence('68:9E:19:07:DE:C3')
    sequence.certify('2016-9-14 10:02')
    scroll_tract.write(sequence)
    metadata = scroll_tract.metadata('68:9E:19:07:DE:C3')
    assert metadata['certification'] == pd.to_datetime('2016-9-14 10:02',
                                                       utc=True)


def test_data(scroll_tract, featurelog):
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    scroll_tract.write(input_sequence)
    data = scroll_tract.data('68:9E:19:07:DE:C3')
    assert data == list(input_sequence)[::-1]


def test_xdata(state_scroll_tract, statelog):
    input_sequence = statelog['88:4A:EA:69:35:BD']
    state_scroll_tract.write(input_sequence)
    data = state_scroll_tract.data('88:4A:EA:69:35:BD',
                                   lower='2018-4-15 00:16:30',
                                   upper='2018-4-15 00:16:45')
    assert len(data) == 3


def test_sequence(scroll_tract, featurelog):
    archive = Store['archive']
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    scroll_tract.write(input_sequence)
    subscriber = MagicMock()
    name_property = PropertyMock(return_value='subscriber')
    type(subscriber).name = name_property
    scroll_tract.update_subscriber(subscriber,
                                   '68:9E:19:07:DE:C3',
                                   pd.NaT)
    sequence = scroll_tract.sequence('68:9E:19:07:DE:C3')
    assert sequence.dt_range == input_sequence.dt_range
    assert sequence.certification is pd.NaT
    assert sequence.consumption is pd.NaT
    scroll_tract.update_subscriber(subscriber,
                                   '68:9E:19:07:DE:C3',
                                   '2016-9-14 10:01')
    sequence = scroll_tract.sequence('68:9E:19:07:DE:C3')
    sequence.certify('2016-9-14 10:01:32.99')
    archive[SCROLL].discard('68:9E:19:07:DE:C3', rge.time_range((None, None)))
    scroll_tract.write(sequence)
    sequence = scroll_tract.sequence('68:9E:19:07:DE:C3')
    assert len(sequence) == 1
    archived = archive[SCROLL].query(id='68:9E:19:07:DE:C3').data()
    assert len(archived) == 2


def test_scroll_pipe(scroll_tract, featurelog):
    pipe = nxr.NxPipe(scroll_tract.store)
    pipe.metadata(scroll_tract, '68:9E:19:07:DE:C3')
    pipe.teardown(scroll_tract, '68:9E:19:07:DE:C3')
    pipe.setup(scroll_tract, '68:9E:19:07:DE:C3')
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    pipe.write(scroll_tract, input_sequence)
    subscriber = MagicMock()
    name_property = PropertyMock(return_value='subscriber')
    type(subscriber).name = name_property
    dt = pd.to_datetime('2016-9-14 10:01', utc=True)
    pipe.update_subscriber(scroll_tract, subscriber, '68:9E:19:07:DE:C3', dt)
    pipe.sequence(scroll_tract, '68:9E:19:07:DE:C3')
    metadata, _, _, _, _, sequence = pipe.execute()
    assert isinstance(metadata['lower'], pd.Timestamp)
    assert sequence.data.equals(input_sequence.data)
    assert sequence.metadata['certification'] is pd.NaT
    assert sequence.metadata['consumption'] == dt


def test_scroll_query(scroll_tract, featurelog):
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    scroll_tract.write(input_sequence)
    query = scroll_tract.query(id=DEVICES)
    assert query.schema == RdSchema()
    assert len(list(query.fetch())) == len(input_sequence)
    assert len(list(query.fetch(limit=1))) == 1
    query_data = query.data()
    assert len(query_data) == 2
    assert query_data[0].data.equals(input_sequence.data)
    assert query_data[1].empty


def test_scroll_query_pipe(procstore, scroll_tract, featurelog):
    input_sequence = featurelog['68:9E:19:07:DE:C3']
    scroll_tract.write(input_sequence)
    pipe = nxr.NxPipe(procstore)
    pipe.query(scroll_tract, id=DEVICES)
    sequences = pipe.execute()[0]
    assert len(sequences) == 2
    assert sequences[0].data.equals(input_sequence.data)
    assert sequences[1].empty


def test_write_xscroll(state_scroll_tract, statelog):
    with pytest.raises(TypeError):
        state_scroll_tract.write(statelog)
    input_sequence = statelog['88:4A:EA:69:35:BD']
    state_scroll_tract.write(input_sequence)
    sequence = state_scroll_tract.sequence('88:4A:EA:69:35:BD')
    assert sequence.data.equals(input_sequence.data)
    assert sequence.dt_range == input_sequence.dt_range
    assert sequence.certification is pd.NaT
    assert sequence.consumption is nxtime.MAX


def test_xmetadata(state_scroll_tract, statelog):
    input_sequence = statelog['88:4A:EA:69:35:BD']
    state_scroll_tract.write(input_sequence)
    metadata = state_scroll_tract.metadata('88:4A:EA:69:35:BD')
    assert set(metadata.keys()) == {'lower', 'upper', 'certification'}
    assert metadata['lower'] == input_sequence.dt_range.lower
    assert metadata['upper'] == input_sequence.dt_range.upper
    assert metadata['certification'] is pd.NaT
    subscriber = MagicMock()
    name_property = PropertyMock(return_value='subscriber')
    type(subscriber).name = name_property
    state_scroll_tract.update_subscriber(subscriber,
                                         '88:4A:EA:69:35:BD',
                                         '2018-4-15 00:16:30')
    metadata = state_scroll_tract.metadata('88:4A:EA:69:35:BD')
    assert set(metadata.keys()) == {'lower', 'upper', 'certification',
                                    'subscriber'}
    assert metadata['certification'] is pd.NaT
    assert metadata['subscriber'] == pd.to_datetime('2018-4-15 00:16:30',
                                                    utc=True)
    sequence = state_scroll_tract.sequence('88:4A:EA:69:35:BD')
    sequence.certify('2018-4-15 00:16:45')
    state_scroll_tract.write(sequence)
    metadata = state_scroll_tract.metadata('88:4A:EA:69:35:BD')
    assert metadata['certification'] == pd.to_datetime('2018-4-15 00:16:45',
                                                       utc=True)


def test_xsequence(state_scroll_tract, archive, statelog):
    input_sequence = statelog['88:4A:EA:69:35:BD']
    state_scroll_tract.write(input_sequence)
    subscriber = MagicMock()
    name_property = PropertyMock(return_value='subscriber')
    type(subscriber).name = name_property
    state_scroll_tract.update_subscriber(subscriber,
                                         '88:4A:EA:69:35:BD',
                                         pd.NaT)
    sequence = state_scroll_tract.sequence('88:4A:EA:69:35:BD')
    assert sequence.dt_range == input_sequence.dt_range
    assert sequence.certification is pd.NaT
    assert sequence.consumption is pd.NaT
    state_scroll_tract.update_subscriber(subscriber,
                                         '88:4A:EA:69:35:BD',
                                         '2018-4-15 00:16:30')
    sequence = state_scroll_tract.sequence('88:4A:EA:69:35:BD')
    sequence.certify('2018-4-15 00:16:45')
    state_scroll_tract.write(sequence)
    sequence = state_scroll_tract.sequence('88:4A:EA:69:35:BD')
    assert len(sequence) == 5
    archived = archive[STATE_SCROLL].query(id='88:4A:EA:69:35:BD').data()
    assert len(archived) == 8
    state_scroll_tract.teardown('88:4A:EA:69:35:BD')
    archived = archive[STATE_SCROLL].query(id='88:4A:EA:69:35:BD').data()
    record = archived[-1]
    assert record.datetime == sequence.certification
    assert record.label is None


def test_xscroll_query(state_scroll_tract, statelog):
    input_sequence = statelog['88:4A:EA:69:35:BD']
    state_scroll_tract.write(input_sequence)
    query = state_scroll_tract.query(id=IDS)
    assert query.schema == StateSchema()
    assert len(list(query.fetch())) == len(input_sequence)
    assert len(list(query.fetch(limit=1))) == 1
    query_data = query.data()
    assert len(query_data) == 2
    assert query_data[1].empty
    assert query_data[0].data.equals(input_sequence.data)


def test_migrate(procstore, backupstore):
    procstore.migrate(backupstore)
    sequence = backupstore['scroll'].sequence('68:9E:19:07:DE:C3')
    assert len(sequence) == 3


def test_load_from_archive(featurelog, statelog,
                           app_full_tract, app_state_tract, appstore,
                           full_tract, state_tract, archive):
    for id in FEATURE_IDS:
        app_full_tract.setup(id, '2016-09-14 10:05:30')
    sequence = app_full_tract.sequence('88:4A:EA:69:DF:A2')
    assert len(sequence) == 5


def test_update_app(featurelog, statelog,
                    app_full_tract, app_state_tract, appstore,
                    full_tract, state_tract, archive):
    for id in FEATURE_IDS:
        app_full_tract.setup(id, '2016-09-14 10:02:00')
    sequence = app_full_tract.sequence('88:4A:EA:69:DF:A2')
    assert len(sequence) == 11
    update_range = slice('2016-9-14 10:02:00', '2016-9-14 10:03:00')
    update = featurelog['88:4A:EA:69:DF:A2'][update_range]
    app_full_tract.update('88:4A:EA:69:DF:A2', update)
    sequence = app_full_tract.sequence('88:4A:EA:69:DF:A2')
    assert len(sequence) == 7


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
