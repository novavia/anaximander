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
from anaximander3.transforms import tasks as tsk, jobs


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
        self.id = id

    def __repr__(self):
        return fun.iformat('id')(self)


D0 = Device('68:9E:19:07:DE:C3')
D1 = Device('88:4A:EA:69:DF:A2')
D2 = Device('88:4A:EA:69:35:BD')
D3 = Device('88:4A:EA:69:38:1A')


class Machine:

    def __init__(self, id, *devices):
        self.id = id
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
BUFFER = Title('buffer', RdSchema)
STATEBUF = Title('statebuf', StateSchema)
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
    store = nxr.RedisArchive(role='redarchive',
                             host=ALT_HOST, port=ALT_PORT, password=PWD)
    tracts = [store.tract(FEATURE), store.tract(STATE), store.tract(SESSION),
              store.tract(EVENT), store.tract(CSTATE)]
    pipe = nxr.NxPipe(store)
    for t, log in zip(tracts, [featurelog, statelog, sessionlog,
                               eventseq, cstateseq]):
        t.create(warn=False, overwrite=True, force=True)
        pipe.append(t, log)
    pipe.execute()
    return store


@pytest.fixture(scope="module")
def archive(featurelog, statelog, sessionlog, eventseq, cstateseq):
    """Creates a redis store for testing purposes."""
    store = _archive(featurelog, statelog, sessionlog, eventseq, cstateseq)
    yield store
    cleanup(store)


def _procstore(archive):
    store = nxr.RedisProcessStore(host=HOST, port=PORT, password=PWD,
                                  archive=archive)
    for title in [FEATURE, STATE, SESSION, EVENT, CSTATE]:
        store.tract(title).create(warn=False, overwrite=True, force=True)
    return store


@pytest.fixture(scope="module")
def procstore(archive):
    """Creates a redis process store for testing purposes."""
    store = _procstore(archive)
    yield store
    cleanup(store)


def _appstore(archive):
    store = nxr.RedisApplicationStore(host=ALT_HOST, port=ALT_PORT,
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


class IdentityTask(tsk.TransformTask):
    __etype__ = Device
    features = tsk.TaskInput(FEATURE)
    output = tsk.TaskOutput(FEATURE)

    def __function__(self):
        return self.features


class FeatureAlertsAssessment(tsk.TransformTask):
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
        events = ops.MultiThresholder(self.features, logger=LOGGER,
                                      thresholds=thresholds)()
        sessions = ops.MultiSessionizer(events, max_gap='60s',
                                        expand='1s', logger=LOGGER,
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


class MachineEventAssessment(tsk.TransformTask):
    __etype__ = Machine
    features = tsk.TaskInput(FEATURE, 'devices')
    events = tsk.TaskOutput(EVENT)

    def __function__(self):
        params = {'column': 'accel_y', 'threshold': 8100}
        events = []
        for dev in self.entity.devices:
            op = ops.Thresholder(self.features[dev.id], **params)
            evs = op().data
            evs['label'] = dev.id
            events.append(evs)
        events = pd.concat(events)
        return dtl.DataSequence(events, schema=MultiEventSchema,
                                id_range=self.entity.id,
                                dt_range=self.dt_range)


class DeviceJob(jobs.Job):
    identity = jobs.JobTask(IdentityTask, input_store='redarchive')
    alerts = jobs.JobTask(FeatureAlertsAssessment, input_store='redarchive')
    __etype__ = Device


class DeviceUpdate(jobs.Job):
    alerts = jobs.JobTask(FeatureAlertsAssessment)
    __etype__ = Device


class InsertFeatures(tsk.InsertTask):
    __etype__ = Device
    target = tsk.TaskOutput(FEATURE)
    latency = '3m'


def test_device_job(featurelog, archive, procstore):
    job = DeviceJob(D1, FEATURE_TME, logger=LOGGER)
    job()
    assert len(job.alerts.alerts) == 6


def test_device_update(featurelog, archive, procstore):
    archive[FEATURE].create(warn=False, overwrite=True, force=True)
    archive[CSTATE].create(warn=False, overwrite=True, force=True)
    minute = pd.Timedelta('1min')
    updates = pd.date_range('2016-9-14 10:01', '2016-9-14 10:10',
                            freq='1T', tz='UTC')
    DeviceUpdate.setup_streaming(D1, 'process', 'process')
    for i, t in enumerate(updates):
        insert = InsertFeatures(D1, *featurelog[D1.id][t - minute:t],
                                output_store='process', logger=LOGGER,
                                stream_mode=True, when=t)
        insert()
        update = DeviceUpdate(D1, stream_mode=True, logger=LOGGER)
        update()
    alerts = procstore[CSTATE].query(id=D1.id).data().\
        to_multisession_sequence()
    features = procstore[FEATURE].sequence(D1.id)
    archived_alerts_query = archive[CSTATE].query(id=D1.id,
                                                  datetime=FEATURE_TME)
    archived_alerts = archived_alerts_query.data().to_multisession_sequence()
    archived_features_query = archive[FEATURE].query(id=D1.id,
                                                     datetime=FEATURE_TME)
    archived_features = archived_features_query.data()
    assert alerts.empty
    assert features.empty
    assert len(archived_alerts) == 4
    assert len(archived_features) == 17


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
    FEATURELOG = featurelog()
    STATELOG = statelog()
    SESSIONLOG = sessionlog()
    EVENTSEQ = eventseq()
    CSTATESEQ = cstateseq(mstateseq())
    LOGS = (FEATURELOG, STATELOG, SESSIONLOG, EVENTSEQ, CSTATESEQ)
