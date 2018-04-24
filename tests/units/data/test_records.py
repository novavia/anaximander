#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for frame.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from collections import OrderedDict
import os
import re
from unittest import TestCase

import pandas as pd
import pytest

import anaximander3 as nx
from anaximander3.data import nxcolumns as cln, nxschema as sch, \
    records as rec


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
FEATURELOG_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')

MAC_PATTERN = re.compile('^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')


# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture(scope="module")
def feature_record():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(FEATURELOG_PATH)
    dataframe.rename(columns={'timestamp': 'datetime',
                              'device': 'id'}, inplace=True)
    return dataframe.iloc[0]

RECORD = feature_record()
RECORD.datetime = pd.to_datetime(RECORD.datetime, utc=True)


def mac_validator(record, column, value):
    if MAC_PATTERN.match(value):
        return True
    else:
        return f"{value} does not match pattern {MAC_PATTERN.pattern}"


class FeatureSchema(sch.SampleLogsSchema):
    id = cln.String(index='nominal', validate=mac_validator)
    datetime = cln.DateTime('UTC', index='sequential')
    Feature_Value_0 = cln.Float()
    Feature_Value_1 = cln.Float()
    Feature_Value_2 = cln.Float()
    Feature_Value_3 = cln.Float()
    Feature_Value_4 = cln.Float()
    Feature_Value_5 = cln.Float()

    @id.validator
    def second_validation(record, column, value):
        return mac_validator(record, column, value)


def test_record(feature_record):
    record = rec.Record(feature_record, schema=FeatureSchema)
    assert record.data.equals(RECORD)
    

if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
