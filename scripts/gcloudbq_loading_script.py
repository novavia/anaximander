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

from google.cloud import bigquery as bq
import pandas as pd

from anaximander.data import fields, schema as sch, gcloudbq as gbq, data
from anaximander.data.tract import tract

from analytics.dbclients.bqclient import RawQuery


PROJECT_ID = 'infinite-uptime-1232'
DATASET_ID = 'KTFL'
TBNAME = "IU_device_data"

BQ = bq.Client(PROJECT_ID)
DATASET = BQ.dataset(DATASET_ID)

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Test Cases
# =============================================================================


@tract
class DeviceData(sch.Schema):
    MAC_ADDRESS = fields.ReStr(key=True, pattern=MAC_PATTERN)
    Timestamp = fields.MilliTimestamp(key=True, sequential=True)
    Timestamp_Pi = fields.Timestamp()
    Feature_Value_0 = fields.Scalar(data.NxFloat)
    Feature_Value_1 = fields.Scalar(data.NxFloat)
    Feature_Value_2 = fields.Scalar(data.NxFloat)
    Feature_Value_3 = fields.Scalar(data.NxFloat)
    Feature_Value_4 = fields.Scalar(data.NxFloat)
    Feature_Value_5 = fields.Scalar(data.NxFloat)


def frame_multi(data, fieldnames):
    ms = DeviceData.multischema
    data = [{k: v for k, v in zip(fieldnames, d)} for d in data]
    records = ms.load(data).data
    return DeviceData.Frame.from_records(records)


def frame(data, fieldnames):
    ms = DeviceData.schema
    data = [{k: v for k, v in zip(fieldnames, d)} for d in data]
    records = [ms.load(d).data for d in data]
    return DeviceData.Frame.from_records(records)    


def nxframe(data, fieldnames):
    dataframe = pd.DataFrame(data, columns=fieldnames)
    return DeviceData.Frame(dataframe)


if __name__ == '__main__':
#    channel = gbq.BigQueryChannel.from_dataset(DATASET, DeviceData, TBNAME)
    sql = "SELECT * FROM [{p}:{d}.{t}] LIMIT 1000"
    sql = sql.format(p=PROJECT_ID, d=DATASET_ID, t=TBNAME)
#    rawquery = channel.rawquery(sql)
#    frame_from_rawquery = rawquery()
#    query = channel.query()
#    frame_from_query = query()
    q = RawQuery(sql)
    data = q.all()
#    ms = DeviceData.Schema(many=True)
#    r = [DeviceData.schema.load({k: v for k, v in zip(q.fieldnames, d)}).data for d in data]
    f = frame(data, q.fieldnames)
    fm = frame_multi(data, q.fieldnames)
    fx = nxframe(data, q.fieldnames)
    