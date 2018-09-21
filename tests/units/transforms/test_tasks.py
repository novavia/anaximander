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

import io
import logging
import os.path

import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.utilities import functions as fun
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl
from anaximander3.io.store import Title
from anaximander3.io import redis as nxr
from anaximander3.transforms import operations as ops
from anaximander3.transforms import tasks as tsk


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
DEVICES = ['88:4A:EA:69:DF:A2', '68:9E:19:07:DE:C3']
IDS = ['88:4A:EA:69:35:BD', '88:4A:EA:69:38:1A']
STATES_TME = ['2018-4-15 00:16:00', '2018-4-15 00:17:00']
EVENTS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
SESSIONS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
MULTIEVENTS = pd.read_csv(MULTIEVENTS_PATH)
MULTISESSIONS = pd.read_csv(MULTISESSIONS_PATH)

LOGGER = logging.Logger('test', level=logging.DEBUG)
LOG_FORMAT = "{asctime} - {levelname} - {name}: {message}"
FORMATTER = logging.Formatter(LOG_FORMAT, style='{')
LOG_FILE = io.StringIO()
FILE_HANDLER = logging.StreamHandler(LOG_FILE)
FILE_HANDLER.setFormatter(FORMATTER)
FILE_HANDLER.setLevel(logging.DEBUG)
CONSOLE_HANDLER = logging.StreamHandler()
CONSOLE_HANDLER.setFormatter(FORMATTER)
CONSOLE_HANDLER.setLevel(logging.DEBUG)
LOGGER.addHandler(FILE_HANDLER)
LOGGER.addHandler(CONSOLE_HANDLER)

# =============================================================================
# Environment
# =============================================================================


class Device:

    def __init__(self, id):
        self.store_id = str(id)

    def __repr__(self):
        return fun.iformat('store_id')(self)


D0 = Device('68:9E:19:07:DE:C3')
D1 = Device('88:4A:EA:69:DF:A2')
D2 = Device('88:4A:EA:69:35:BD')
D3 = Device('88:4A:EA:69:38:1A')


class Machine:

    def __init__(self, id, *devices):
        self.store_id = str(id)
        self.devices = devices


M0 = Machine('1', D0, D1)


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


FEATURE = Title('feature', RdSchema)
STATE = Title('state', StateSchema)
SESSION = Title('session', SessionSchema)
EVENT = Title('event', MultiEventSchema)
CSTATE = Title('cstate', sch.CompoundStateLogsSchema)
SCROLL = Title('scroll', RdSchema)
STATE_SCROLL = Title('state_scroll', StateSchema)
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


def _archive(featurelog, statelog, sessionlog, eventseq, cstateseq):
    """Primitive for archive."""
    import time
    t0 = time.time()
    store = nxr.RedisArchive(role='archive',
                             host=HOST, port=PORT, password=PWD)
    t1 = time.time()
    runtime_ms = round(1e3 * (t1 - t0))
    print(f"Instantiated store in {runtime_ms:,} milliseconds.")
    tracts = [store.tract(FEATURE), store.tract(STATE), store.tract(SESSION),
              store.tract(EVENT), store.tract(CSTATE)]
    t2 = time.time()
    runtime_ms = round(1e3 * (t2 - t1))
    print(f"Instantiated tracts in {runtime_ms:,} milliseconds.")
    pipe = nxr.NxPipe(store)
    for t, log in zip(tracts, [featurelog, statelog, sessionlog,
                               eventseq, cstateseq]):
        t.create(warn=False, overwrite=True, force=True)
        pipe.append(t, log)
    t3 = time.time()
    runtime_ms = round(1e3 * (t3 - t2))
    print(f"Created tracts in {runtime_ms:,} milliseconds.")
    pipe.execute()
    t4 = time.time()
    runtime_ms = round(1e3 * (t4 - t3))
    print(f"Uploaded logs in {runtime_ms:,} milliseconds.")
    return store


@pytest.fixture(scope="module")
def archive(featurelog, statelog, sessionlog, eventseq, cstateseq):
    """Creates a redis store for testing purposes."""
    store = _archive(featurelog, statelog, sessionlog, eventseq, cstateseq)
    yield store
    cleanup(store)


def _procstore(archive):
    return nxr.RedisPipeline(host=HOST, port=PORT, password=PWD,
                             archive=archive)


@pytest.fixture(scope="module")
def procstore(archive):
    """Creates a redis pipeline store for testing purposes."""
    store = _procstore(archive)
    yield store
    cleanup(store)


def _appstore(archive):
    store = nxr.RedisBuffer(host=ALT_HOST, port=ALT_PORT,
                            password=PWD, archive=archive)
    feature_tract = store.tract(FEATURE, depth='1h')
    feature_tract.create(warn=False, overwrite=True, force=True)
    return store


@pytest.fixture(scope="module")
def appstore(archive):
    """Creates a redis application store for testing purposes."""
    store = _appstore(archive)
    yield store
    cleanup(store)

# =============================================================================
# Test Cases
# =============================================================================

# Specifies that tests are skipped if tester is not online.
pytestmark = [pytest.mark.online, pytest.mark.gcloud]


class IdentityTask(tsk.Task):
    __etype__ = Device
    features = tsk.TaskInput(FEATURE)
    output = tsk.TaskOutput(FEATURE)

    def __function__(self):
        return self.features


class FeatureAlertsAssessment(tsk.Task):
    __etype__ = Device
    features = tsk.TaskInput(FEATURE)
    alerts = tsk.TaskOutput(CSTATE)

    def __function__(self):
        thresholds = {'accel_x': 1000,
                      'accel_y': 8100}
        onset_spans = {}
        if self.alerts is not None:
            alert_spans = self.alerts.to_multisession_sequence()
            certified_spans = alert_spans[None:self.alerts.certification]
            for l in certified_spans.label.categories:
                try:
                    span = certified_spans.session_sequence(l)[-1].span
                except IndexError:
                    continue
                else:
                    onset_spans[l] = span
        events = ops.MultiThresholder(self.features, logger=self.logger,
                                      thresholds=thresholds)()
        sessions = ops.MultiSessionizer(events, max_gap='75s',
                                        expand='1s', logger=self.logger,
                                        onset_spans=onset_spans)()
        return sessions.to_compound_state_sequence()

    def plot(self):
        sessions = self.alerts.to_multisession_sequence()
        score = self.features.plot(certificate=False)
        thresholds = {'accel_x': 1000,
                      'accel_y': 8100}
        for f, staff in zip(self.features.features, score.staves):
            staff.ax.axhline(thresholds.get(f, 0), color='red')
            label = 'peaking_' + f
            sessions.session_sequence(label).plot(staff=staff, make_main=True)


class MachineEventAssessment(tsk.Task):
    __etype__ = Machine
    features = tsk.TaskInput(FEATURE, 'devices')
    events = tsk.TaskOutput(EVENT)

    def __function__(self):
        params = {'column': 'accel_y', 'threshold': 8100}
        events = []
        for dev in self.entity.devices:
            op = ops.Thresholder(self.features[dev.store_id], **params)
            evs = op().data
            evs['label'] = dev.store_id
            events.append(evs)
        events = pd.concat(events)
        return dtl.DataSequence(events, schema=MultiEventSchema,
                                id_range=self.entity.store_id,
                                dt_range=self.dt_range)


class UpdateTask(tsk.UpdateTask):
    __etype__ = Device
    features = tsk.TaskInput(FEATURE)
    output = tsk.TaskOutput(FEATURE)


def test_task_input():
    ti_0 = tsk.TaskInput(FEATURE, 'devices')
    ti_1 = tsk.TaskInput(FEATURE, 'devices')
    ALT_FEATURE = Title('feature', RdSchema)
    ti_2 = tsk.TaskInput(ALT_FEATURE, 'devices')
    s = {ti_0, ti_1, ti_2}
    assert len(s) == 1


def test_nothing_task(archive, featurelog):
    task = IdentityTask(D0, None, archive, logger=LOGGER)
    output, *_ = task()
    assert isinstance(output, dtl.DataSequence)
    assert output.empty
    task = IdentityTask(D0, FEATURE_TME, archive, logger=LOGGER)
    output, *_ = task()
    assert output == featurelog[D0.store_id]


def test_feature_alerts(archive):
    task = FeatureAlertsAssessment(D1, FEATURE_TME, 'archive', logger=LOGGER)
    output, *_ = task()
    assert len(output.to_multisession_sequence()) == 4


def test_machine_events(archive):
    task = MachineEventAssessment(M0, FEATURE_TME, 'archive', logger=LOGGER)
    output, *_ = task()
    assert len(output) == 7


def test_update_task(archive, featurelog, procstore, appstore):
    tract = procstore.tract(FEATURE)
    tract.create(warn=False, overwrite=True, force=True)
    featurelog.certify('2018-9-14 10:05')
    tract.write(featurelog[D0.store_id])
    UpdateTask.setup_streaming(D0, when='2016-9-14 10:00')
    task = UpdateTask(D0, logger=LOGGER, stream_mode=True)
    output, *_ = task()
    assert output.data.equals(featurelog[D0.store_id].data)
    query = appstore[FEATURE].query(id=D0.store_id, datetime=FEATURE_TME)
    assert query.data() == output


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
