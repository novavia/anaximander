#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script module for frames.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import os

import pandas as pd

import anaximander as nx
from anaximander.data import schema, fields
from anaximander.data.tract import tract


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

MAC = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Test Cases
# =============================================================================


def featurelog():
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


@tract
class FeatureSchema(schema.Schema):
    device = fields.ReStr(MAC, key=True)
    timestamp = fields.DateTime(key=True, sequential=True)
    Feature_Value_0 = fields.Float()
    Feature_Value_1 = fields.Float()
    Feature_Value_2 = fields.Float()
    Feature_Value_3 = fields.Float()
    Feature_Value_4 = fields.Float()
    Feature_Value_5 = fields.Float()


if __name__ == '__main__':
    DATA = pd.read_csv(LOGFILE_PATH)
    DATA.timestamp = pd.to_datetime(DATA.timestamp)
