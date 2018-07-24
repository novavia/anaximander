#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for operations.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import io
import logging
import os
import re

import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    datalogs as dtl
from anaximander3.transforms import operations as opr


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
SAMPLES_PATH = os.path.join(TEST_DATA_DIR, 'samples.csv')
EVENTS_PATH = os.path.join(TEST_DATA_DIR, 'events.csv')
STATES_PATH = os.path.join(TEST_DATA_DIR, 'states.csv')
SESSIONS_PATH = os.path.join(TEST_DATA_DIR, 'sessions.csv')
PERIODS_PATH = os.path.join(TEST_DATA_DIR, 'periods.csv')

SAMPLES = pd.read_csv(SAMPLES_PATH)
EVENTS = pd.read_csv(EVENTS_PATH)
STATES = pd.read_csv(STATES_PATH)
SESSIONS = pd.read_csv(SESSIONS_PATH)
PERIODS = pd.read_csv(PERIODS_PATH)

IDS = ['88:4A:EA:69:35:BD', '88:4A:EA:69:38:1A']
SAMPLES_TME = ['2018-4-15 00:15:00', '2018-4-15 00:15:10']
EVENTS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
STATES_TME = ['2018-4-15 00:16:00', '2018-4-15 00:17:00']
SESSIONS_TME = ['2018-4-15 00:15:00', '2018-4-15 00:18:00']
PERIODS_TME = ['2018-4-14 00:00:00', '2018-4-15 00:15:00']

DUMP_PATH = os.path.join(TEST_DATA_DIR, 'dump.csv')

MAC_PATTERN = re.compile('^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')

LOGGER = logging.Logger('test', level=logging.DEBUG)
LOG_FORMAT = "{asctime} - {levelname} - {name}: {message}"
FORMATTER = logging.Formatter(LOG_FORMAT, style='{')
LOG_FILE = io.StringIO()
FILE_HANDLER = logging.StreamHandler(LOG_FILE)
FILE_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(FILE_HANDLER)


class SampleSchema(sch.SampleLogsSchema):
    id = cln.String(index='nominal')
    datetime = cln.DateTime(tz='UTC', index='sequential')
    accel_energy_512 = cln.Float()
    temperature = cln.Float()
    charge = cln.Integer()

    @id.validator
    def mac_validator(record, column, value):
        if MAC_PATTERN.match(value):
            return True
        else:
            return f"{value} does not match pattern {MAC_PATTERN.pattern}"


class EventSchema(sch.EventLogsSchema):
    label = cln.EventLabel(('heartbeat',))
    message = cln.Text()
    latency = cln.Float()


class StateSchema(sch.StateLogsSchema):
    label = cln.StateLabel(('Loading', 'Executing'))


class SessionSchema(sch.SessionLogsSchema):
    part_id = cln.Integer()


class PeriodSchema(sch.PeriodLogsSchema):
    freq = '5T'
    duration = cln.Float()
    data_coverage = cln.Percentage()
    part_count = cln.Integer()
    avg_cycle = cln.Float()
    oee = cln.Percentage()
    quota = cln.Bool()


class GenericSchema(sch.SampleLogsSchema):
    feature = cln.Float(generic=True)


@pytest.fixture(scope="module")
def samples():
    return dtl.DataLog(SAMPLES, schema=SampleSchema, id_range=IDS,
                       dt_range=SAMPLES_TME)


@pytest.fixture(scope="module")
def events():
    return dtl.DataLog(EVENTS, schema=EventSchema, id_range=IDS,
                       dt_range=EVENTS_TME)


@pytest.fixture(scope="module")
def states():
    return dtl.DataLog(STATES, schema=StateSchema, id_range=IDS,
                       dt_range=STATES_TME)


@pytest.fixture(scope="module")
def sessions():
    return dtl.DataLog(SESSIONS, schema=SessionSchema, id_range=IDS,
                       dt_range=SESSIONS_TME)


@pytest.fixture(scope="module")
def periods():
    return dtl.DataLog(PERIODS, schema=PeriodSchema, id_range=IDS,
                       dt_range=PERIODS_TME)

# =============================================================================
# Test Cases
# =============================================================================


def test_identity(samples):
    op = opr.LoggerIdentity(samples, logger=LOGGER)
    assert op() == samples
    assert 'Processing' in LOG_FILE.getvalue()
    with pytest.raises(opr.InputError):
        opr.Identity(None, logger=LOGGER)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
