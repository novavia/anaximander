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

import os
import pandas as pd
import pytz

from google.cloud import bigtable
from grpc import RpcError

from anaximander.utilities.functions import get
from anaximander.data import fields, schema as sch, data
from anaximander.data.tract import tract, DataDomain, domain


PROJECT_ID = 'infinite-uptime-1232'

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = SCRIPT_DIR
DATA_FILE = os.path.join(DATA_DIR, 'sampledata.csv')
INSTANCE_ID = 'testinstance'

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

BTCLIENT = bigtable.Client(project=PROJECT_ID, admin=True)
INSTANCE = BTCLIENT.instance(INSTANCE_ID)
TABLE = INSTANCE.table('DeviceData')

MAXTIME = 4102444800.0  # Unix Timestamp of 1-1-2100, UTC

DEVICES = ['88:4A:EA:69:42:9B', '88:4A:EA:69:43:8A', '88:4A:EA:69:E1:72']

# =============================================================================
# Test Cases
# =============================================================================

datadomain = DataDomain('iudevice')


@domain('iudevice')
@tract
class DeviceData(sch.Schema):
    device = fields.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fields.Timestamp(key=True, sequential=True)
    Feature_Value_0 = fields.Scalar(data.NxFloat)
    Feature_Value_1 = fields.Scalar(data.NxFloat)
    Feature_Value_2 = fields.Scalar(data.NxFloat)
    Feature_Value_3 = fields.Scalar(data.NxFloat)
    Feature_Value_4 = fields.Scalar(data.NxFloat)
    Feature_Value_5 = fields.Scalar(data.NxFloat)


def create_table(raise_on_error=False):
    try:
        TABLE.create()
    except RpcError:
        if raise_on_error:
            raise
        else:
            return
    features = TABLE.column_family('features')
    features.create()


def delete_table():
    try:
        TABLE.delete()
    except RpcError:
        return


def make_key(device, timestamp):
    device = device[::-1]
    if not timestamp.tz:
        timestamp = timestamp.tz_localize(pytz.UTC)
    timestamp = MAXTIME - timestamp.timestamp()
    return '#'.join([device, str(timestamp)])


def unmake_key(key):
    key = key.decode('utf-8')
    device_, timestamp_ = key.split('#')
    device = device_[::-1]
    timestamp = pd.Timestamp((MAXTIME - float(timestamp_)) * 1e9, tz=pytz.UTC)
    return device, timestamp


def insert(record):
    key = make_key(record.device, record.timestamp)
    row = TABLE.row(key)
    for attr in DeviceData.Schema.nonkeyfieldnames:
        row.set_cell('features',
                     attr.encode('utf-8'),
                     str(getattr(record, attr)).encode('utf-8'))
    row.commit()


def upload(frame):
    for record in frame.to_records():
        insert(record)


def read_record(key, row):
    device, timestamp = unmake_key(key)
    features = {k.decode('utf-8'): v[0].value.decode('utf-8')
                for k, v in row.cells['features'].items()}
    return DeviceData.Record(device, timestamp, **features)


class Query:

    def __init__(self, start_time=None, end_time=None, devices=None):
        if devices is None:
            devices = DEVICES
        self.devices = devices
        self.start_time = start_time
        self.end_time = end_time
        
        start_time = get(start_time, '2000-1-1')
        end_time = get(end_time, '2100-1-1')
        start = pd.Timestamp(start_time, tz=pytz.UTC)
        end_keys = [make_key(d, start) for d in devices]
        end = pd.Timestamp(end_time, tz=pytz.UTC)
        start_keys = [make_key(d, end) for d in devices]
        self.keys = list(zip(start_keys, end_keys))

    def fetch(self):
        """A record generator."""
        row_groups = [TABLE.read_rows(s, e) for s, e in self.keys]
        for g in row_groups:
            g.consume_all()
            for key, row in g.rows.items():
                yield read_record(key, row)

    def to_frame(self):
        """Returns a frame."""
        return DeviceData.Frame.from_records(self.fetch())


if __name__ == '__main__':
    frame = DeviceData.Frame.from_csv(DATA_FILE)
    create_table()
    upload(frame)
    query = Query()
    print(query.to_frame().data)
    delete_table()
    