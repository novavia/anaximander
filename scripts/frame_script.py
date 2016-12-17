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

import numpy as np
import pandas as pd

import anaximander as nx
from anaximander.data import frame, schema


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
LOGFILE_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

# =============================================================================
# Test Cases
# =============================================================================


def featurelog():
    dataframe = pd.read_csv(LOGFILE_PATH)
    dataframe.timestamp = pd.to_datetime(dataframe.timestamp)
    return dataframe


class FeatureSchema(schema.Schema):
    device = schema.Str(key=True)
    timestamp = schema.DateTime(key=True, sequential=True)
    Feature_Value_0 = schema.Float()
    Feature_Value_1 = schema.Float()
    Feature_Value_2 = schema.Float()
    Feature_Value_3 = schema.Float()
    Feature_Value_4 = schema.Float()
    Feature_Value_5 = schema.Float()


class FeatureFrame(frame.Frame):
    pass

FeatureFrame.schema = FeatureSchema


if __name__ == '__main__':
    DATA = pd.read_csv(LOGFILE_PATH)
    DATA.timestamp = pd.to_datetime(DATA.timestamp)
