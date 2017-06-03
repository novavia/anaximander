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

from contextlib import contextmanager
import os.path

from google.cloud import bigquery as bq
import numpy as np
import pandas as pd

import anaximander as nx
from anaximander.utilities.functions import get_gcloud_project_id
from anaximander.data import data as dat, fields as fld, schema as sch, \
    tract as trc, gcbigquery as gbq


PROJECT_ID = get_gcloud_project_id()
BQ = bq.Client(PROJECT_ID)
DATASET_ID = 'InterfaceTesting'

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')


# =============================================================================
# Domain declaration
# =============================================================================


test_domain = trc.DataDomain('test')


@trc.domain('test')
@trc.tract
class DeviceData(sch.Schema):
    mac = fld.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fld.DateTime(key=True, sequential=True)
    accel_x = fld.Scalar(dat.NxFloat)

# =============================================================================
# Functions
# =============================================================================


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


@contextmanager
def session(clean=True):
    dataset = BQ.dataset(DATASET_ID)
    if dataset.exists():
        cleanup(dataset)
    else:
        dataset.crate()
    gbq.create_all(dataset, test_domain)
    yield dataset  # provide the fixture value
    if clean:
        cleanup(dataset)

# =============================================================================
# Main
# =============================================================================


if __name__ == '__main__':
    data = DeviceData.Frame(featurelog())
    records = data.to_records()
    with session(False) as dataset:
        channel = gbq.BigQueryChannel.from_dataset(dataset, DeviceData)
        response = channel.append(data)
    query = channel.query()
    data_load = query()
    ax = data.data.accel_x.values
    ax_load = data_load.data.accel_x.values
    assert np.array_equal(ax, ax_load)
    response = channel.insert(*records)
    query = channel.query()
    data_load = query()
    assert len(data_load) == 40
