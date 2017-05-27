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

from anaximander.data import fields, schema as sch, data
from anaximander.data.tract import tract
from anaximander.data import gcbigquery as gbq
from anaximander.utilities.functions import get_gcloud_project_id


PROJECT_ID = get_gcloud_project_id()
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
    Timestamp = fields.Timestamp(key=True, sequential=True)
    Timestamp_Pi = fields.Timestamp()
    Feature_Value_0 = fields.Scalar(data.NxFloat)


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
    sql = "SELECT MAC_ADDRESS, Timestamp, Timestamp_Pi, Feature_Value_0 \
           FROM [{p}:{d}.{t}] LIMIT 1000"
    sql = sql.format(p=PROJECT_ID, d=DATASET_ID, t=TBNAME)
    schema = DeviceData.Schema
    table = gbq.BigQueryDataTable[schema](DATASET, TBNAME)
    q = table.query(sql=sql)
    data = list(q.fetch())
    f = frame(data, q.fields)
    fm = frame_multi(data, q.fields)
    fx = nxframe(data, q.fields)
    
