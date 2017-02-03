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

from gcloud import bigquery as bq

from anaximander.data import fields, schema as sch, gcloudbq as gbq, data
from anaximander.data.tract import tract


PROJECT_ID = 'infinite-uptime-1232'
DATASET_ID = 'FORGEMAX2'
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
    Timestamp = fields.DateTime(key=True, sequential=True)
    Feature_Value_0 = fields.Scalar(data.NxFloat)
    Feature_Value_1 = fields.Scalar(data.NxFloat)
    Feature_Value_2 = fields.Scalar(data.NxFloat)
    Feature_Value_3 = fields.Scalar(data.NxFloat)
    Feature_Value_4 = fields.Scalar(data.NxFloat)
    Feature_Value_5 = fields.Scalar(data.NxFloat)


if __name__ == '__main__':
    channel = gbq.BigQueryChannel.from_dataset(DATASET, DeviceData, TBNAME)
    sql = "SELECT * FROM [{p}:{d}.{t}] LIMIT 10"
    sql = sql.format(p=PROJECT_ID, d=DATASET_ID, t=TBNAME)
    rawquery = channel.rawquery(sql)
    frame_from_rawquery = rawquery()
    query = channel.query()
    frame_from_query = query()
