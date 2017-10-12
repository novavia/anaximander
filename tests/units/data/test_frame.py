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
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

import anaximander as nx
from anaximander.utilities.nxtime import datetime, tz_aware, tz_naive
from anaximander.data import fields, frame as frm, schema
from anaximander.data import NxFloat, NxRecord, NxSeries


NXPATH = os.path.dirname(nx.__path__[0])
TEST_DATA_DIR = os.path.join(NXPATH, 'tests/data')
FEATURELOG_PATH = os.path.join(TEST_DATA_DIR, 'featurelog.csv')
FEATURELOG_PARTIAL_PATH = os.path.join(TEST_DATA_DIR, 'featurelog_partial.csv')
FEATURELOG_NO_DEVICE_PATH = os.path.join(TEST_DATA_DIR,
                                         'featurelog_no_device.csv')
FEATURELOG_SHUFFLED_PATH = os.path.join(TEST_DATA_DIR,
                                        'featurelog_shuffled.csv')
FEATURELOG_SUPERFLUOUS_PATH = os.path.join(TEST_DATA_DIR,
                                           'featurelog_superfluous.csv')
FEATURELOG_INVALID_PATH = os.path.join(TEST_DATA_DIR, 'featurelog_invalid.csv')

DUMP_PATH = os.path.join(TEST_DATA_DIR, 'dump.csv')

MAC_PATTERN = '^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture(scope="module")
def featurelog():
    """Returns a nominal dataframe."""
    dataframe = pd.read_csv(FEATURELOG_PATH)
    dataframe.timestamp = dataframe.timestamp.apply(datetime)
    return dataframe

LOG = featurelog()


@pytest.fixture(scope="module")
def featurelog_partial():
    """Returns a dataframe with missing columns."""
    dataframe = pd.read_csv(FEATURELOG_PARTIAL_PATH)
    dataframe.timestamp = dataframe.timestamp.apply(datetime)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_no_device():
    """Returns a dataframe missing a key column."""
    dataframe = pd.read_csv(FEATURELOG_NO_DEVICE_PATH)
    dataframe.timestamp = dataframe.timestamp.apply(datetime)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_shuffled():
    """Returns a dataframe with columns shuffled."""
    dataframe = pd.read_csv(FEATURELOG_SHUFFLED_PATH)
    dataframe.timestamp = dataframe.timestamp.apply(datetime)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_superfluous():
    """Returns a dataframe with an extra column not in the schema."""
    dataframe = pd.read_csv(FEATURELOG_SUPERFLUOUS_PATH)
    dataframe.timestamp = dataframe.timestamp.apply(datetime)
    return dataframe


@pytest.fixture(scope="module")
def featurelog_invalid():
    """Returns a dataframe with invalid values."""
    dataframe = pd.read_csv(FEATURELOG_INVALID_PATH)
    dataframe.timestamp = dataframe.timestamp.apply(datetime)
    return dataframe


class FeatureSchema(schema.Schema):
    device = fields.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fields.Timestamp(key=True, sequential=True)
    Feature_Value_0 = fields.Scalar(NxFloat)
    Feature_Value_1 = fields.Scalar(NxFloat)
    Feature_Value_2 = fields.Scalar(NxFloat)
    Feature_Value_3 = fields.Scalar(NxFloat)
    Feature_Value_4 = fields.Scalar(NxFloat)
    Feature_Value_5 = fields.Scalar(NxFloat)


class FeatureSchemaMultKeys(schema.Schema):
    device = fields.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fields.Timestamp(key=True, sequential=True)
    Feature_Value_0 = fields.Scalar(NxFloat, key=True)
    Feature_Value_1 = fields.Scalar(NxFloat)
    Feature_Value_2 = fields.Scalar(NxFloat)


class FeatureSchemaNoKey(schema.Schema):
    device = fields.ReStr(pattern=MAC_PATTERN)
    timestamp = fields.Timestamp(sequential=True)
    Feature_Value_0 = fields.Scalar(NxFloat)
    Feature_Value_1 = fields.Scalar(NxFloat)
    Feature_Value_2 = fields.Scalar(NxFloat)
    Feature_Value_3 = fields.Scalar(NxFloat)
    Feature_Value_4 = fields.Scalar(NxFloat)
    Feature_Value_5 = fields.Scalar(NxFloat)


class FeatureSchemaNoSequentialKey(schema.Schema):
    device = fields.ReStr(key=True, pattern=MAC_PATTERN)
    timestamp = fields.Timestamp(sequential=True)
    Feature_Value_0 = fields.Scalar(NxFloat)
    Feature_Value_1 = fields.Scalar(NxFloat)
    Feature_Value_2 = fields.Scalar(NxFloat)


class FeatureSchemaSequentialKey(schema.Schema):
    timestamp = fields.Timestamp(key=True, sequential=True)
    Feature_Value_0 = fields.Scalar(NxFloat)
    Feature_Value_1 = fields.Scalar(NxFloat)
    Feature_Value_2 = fields.Scalar(NxFloat)


class FeatureMapping(frm.NxDataMapping):
    schema = FeatureSchema


class FeatureMappingMK(frm.NxDataMapping):
    schema = FeatureSchemaMultKeys


class FeatureMappingNK(frm.NxDataMapping):
    schema = FeatureSchemaNoKey


class FeatureCollectionNK(frm.NxDataCollection):
    schema = FeatureSchemaNoKey


class FeatureMappingNS(frm.NxDataMapping):
    schema = FeatureSchemaNoSequentialKey


class FeatureMappingSK(frm.NxDataMapping):
    schema = FeatureSchemaSequentialKey


class FeatureSequence(frm.NxDataSequence):
    schema = FeatureSchema


class FeatureSequenceMK(frm.NxDataSequence):
    schema = FeatureSchemaMultKeys


class FeatureSequenceNK(frm.NxDataSequence):
    schema = FeatureSchemaNoKey


class FeatureSequenceNS(frm.NxDataSequence):
    schema = FeatureSchemaNoSequentialKey


class FeatureSequenceSK(frm.NxDataSequence):
    schema = FeatureSchemaSequentialKey


def test_dtypes():
    """Tests the dtypes function."""
    spec = OrderedDict([('device', np.dtype('object')),
                        ('timestamp', np.dtype('datetime64[ns]')),
                        ('Feature_Value_0', np.dtype('float')),
                        ('Feature_Value_1', np.dtype('float')),
                        ('Feature_Value_2', np.dtype('float')),
                        ('Feature_Value_3', np.dtype('float')),
                        ('Feature_Value_4', np.dtype('float')),
                        ('Feature_Value_5', np.dtype('float'))])
    assert frm.dtypes(FeatureSchema) == spec


class TestFrame(TestCase):

    def test_empty(self):
        assert FeatureMapping().empty

    def test_cast(self):
        native_log = LOG.astype({'timestamp': np.dtype('object')})
        cast = FeatureMappingNK.cast(native_log)
        # Casts converts the timestamp back to datetime
        assert cast.equals(LOG)
        assert not cast.equals(native_log)

    def test_cast_partial(self):
        """Tests with a dataframe missing non-essential columns."""
        log = featurelog_partial()
        cast = FeatureMappingNK.cast(log)
        assert cast.equals(log)

    def test_cast_no_device(self):
        """Tests with a dataframe missing key column 'device'."""
        log = featurelog_no_device()
        with pytest.raises(frm.ConformityError):
            FeatureMapping.cast(log)

    def test_cast_shuffled(self):
        """Tests with a dataframe whose columns are out of sequence."""
        log = featurelog_shuffled()
        cast = FeatureMappingNK.cast(log)
        assert frm.dtypes(FeatureSchema) == OrderedDict(tz_naive(cast).dtypes)

    def test_cast_superfluous(self):
        """Tests with a dataframe with a superfluous column."""
        log = featurelog_superfluous()
        cast = FeatureMappingNK.cast(log)
        assert len(log.columns) == len(cast.columns) + 1
        assert frm.dtypes(FeatureSchema) == OrderedDict(tz_naive(cast).dtypes)

    def test_init(self):
        frame = FeatureMappingNK(LOG)
        assert frame.data.equals(LOG)

    def test_validate(self):
        frame = FeatureMappingNK(LOG, validate=True)
        assert frame.data.equals(LOG)
        invalid_log = featurelog_invalid()
        with pytest.raises(schema.ValidationError):
            frame = FeatureMappingNK(invalid_log, validate=True)


class TestMapping(TestCase):

    def test_properties(self):
        log = FeatureMapping(LOG)
        assert log.sqcol == 'timestamp'
        assert log.kcols == ['device']
        assert isinstance(log.index, pd.DatetimeIndex)

    def test_getitem(self):
        log = FeatureMapping(LOG)
        devices = list(np.unique(log.data.device))
        assert isinstance(log[devices], FeatureMapping)
        assert log[devices].data.equals(log.data)
        assert isinstance(log[devices[0]], FeatureSequence)
        assert len(log[devices[0]]) == 3
        with pytest.raises(KeyError):
            log[0]

    def test_iter(self):
        log = FeatureMapping(LOG)
        devices = list(np.unique(log.data.device))
        keys = [list(log.keys())[i][0] for i in [0, 1]]
        assert set(keys) == set(devices)

    def test_getitem_mk(self):
        log = FeatureMappingMK(LOG)
        devices = list(np.unique(log.data.device))
        assert isinstance(log[devices[0]], FeatureMappingMK)
        assert len(log[devices[0]]) == 3
        v0 = log.data.Feature_Value_0[0]
        assert isinstance(log[devices[0], v0], FeatureSequenceMK)
        assert isinstance(log[devices[0], v0][0], NxRecord)

    def test_getitem_nk(self):
        log = FeatureMappingNK(LOG)
        assert isinstance(log[0], NxRecord)

    def test_iter_nk(self):
        log = FeatureMappingNK(LOG)
        keys = set(log.keys())
        assert keys == set(range(20))
        values = set(log.values())
        assert values == {NxRecord[FeatureSchemaNoKey](**log.data.ix[i])
                          for i in range(20)}

    def test_getitem_ns(self):
        log = FeatureMappingNS(LOG)
        devices = list(np.unique(log.data.device))
        assert isinstance(log[devices[0]], frm.NxDataCollection)
        assert len(log[devices[0]]) == 3

    def test_getitem_sk(self):
        log = FeatureMappingSK(LOG)
        assert isinstance(log[0], NxRecord)

    def test_iloc(self):
        log = FeatureMapping(LOG)
        assert isinstance(log.iloc[0:1], FeatureMapping)
        assert len(log.iloc[0:1]) == 1
        assert isinstance(log.iloc[0], NxRecord)

    def test_loc(self):
        log = FeatureMapping(LOG)
        t0 = log.index[5]
        t1 = log.index[10]
        assert isinstance(log.loc[t0:t1], FeatureMapping)
        assert len(log.loc[t0:t1]) == 6
        assert isinstance(log.loc[t0], NxRecord)

    def test_from_records(self):
        log = FeatureMapping(LOG)
        rtype = NxRecord[FeatureSchema]
        records = [rtype(*LOG.ix[i]) for i in range(len(LOG))]
        log_fr = FeatureMapping.from_records(records)
        assert log.data.equals(log_fr.data)

    def test_to_records(self):
        log = FeatureMapping(LOG)
        rtype = NxRecord[FeatureSchema]
        records = log.to_records()
        assert len(log) == 20
        assert all(isinstance(r, rtype) for r in records)

    def test_extend(self):
        # Extend with pandas.DF
        log = FeatureMapping(LOG)
        log.extend(LOG)
        assert len(log) == 40
        # Extend with NxDF
        log = FeatureMapping(LOG)
        log.extend(log)
        assert len(log) == 40
        # Extend with NxDF that has different but compatible schema
        log_a = FeatureMappingNS(LOG)
        log_b = FeatureMapping(LOG)
        log_a.extend(log_b)
        assert len(log) == 40

    def test_append(self):
        log = FeatureMapping(LOG)
        record = log.iloc[0]
        log.append(record)
        assert len(log) == 21


class TestSequence(TestCase):

    def setUp(self):
        log = FeatureMapping(LOG)
        devices = list(np.unique(log.data.device))
        self.sequence = log[devices[0]]

    def test_getitem(self):
        seq = self.sequence
        assert isinstance(seq[0:1], FeatureSequence)
        assert len(seq[0:1]) == 1
        assert isinstance(seq[0], NxRecord)

    def test_iter(self):
        seq = self.sequence
        assert list(seq) == [NxRecord[FeatureSchema](**seq.data.ix[i])
                             for i in range(3)]

    def test_iloc(self):
        seq = self.sequence
        assert isinstance(seq.iloc[0:1], FeatureSequence)
        assert len(seq.iloc[0:1]) == 1
        assert isinstance(seq.iloc[0], NxRecord)

    def test_loc(self):
        seq = self.sequence
        t0 = seq.index[1]
        assert isinstance(seq.loc[t0:t0], FeatureSequence)
        assert len(seq.loc[t0:t0]) == 1
        assert isinstance(seq.loc[t0], NxRecord)

    def test_verification(self):
        with pytest.raises(frm.ConformityError):
            FeatureSequence(LOG)

    def test_getitem_nk(self):
        seq = FeatureSequenceNK(LOG)
        assert isinstance(seq[0:1], FeatureSequenceNK)
        assert len(seq[0:1]) == 1
        assert isinstance(seq[0], NxRecord)


class TestColumnProxy(TestCase):

    def test_type_assignments(self):
        assert all(isinstance(getattr(FeatureMapping, col), property)
                   for col in FeatureSchema.fields)

    def test_proxy(self):
        log = FeatureMappingNK(LOG)
        assert isinstance(log.device, frm.ColumnProxy)
        assert isinstance(log.device(), pd.Series)
        assert log.device().equals(LOG.device)
        assert isinstance(log.Feature_Value_0(), NxSeries)
        assert log.Feature_Value_0().data.equals(LOG.Feature_Value_0)


class TestLoader(TestCase):

    def test_load_csv(self):
        loader = frm.CsvLoader(FeatureSchemaNoKey, FEATURELOG_PATH)
        frame = loader()
        assert isinstance(frame, FeatureCollectionNK)
        assert frame.data.equals(LOG)
        frame = loader(FeatureMappingNK)
        assert isinstance(frame, FeatureMappingNK)

    def test_from_csv(self):
        frame = FeatureMappingNK.from_csv(FEATURELOG_PATH)
        assert isinstance(frame, FeatureMappingNK)
        assert frame.data.equals(LOG)


class TestDumper(TestCase):

    def test_dump_csv(self):
        frame = FeatureMappingNK(LOG)
        dumper = frm.CsvDumper(frame, DUMP_PATH)
        dumper()
        dataframe = pd.read_csv(DUMP_PATH)
        dataframe.timestamp = dataframe.timestamp.apply(datetime)
        assert dataframe.equals(frame.data)

    def test_to_csv(self):
        frame = FeatureMappingNK(LOG)
        frame.to_csv(DUMP_PATH)
        dataframe = pd.read_csv(DUMP_PATH)
        dataframe.timestamp = dataframe.timestamp.apply(datetime)
        assert dataframe.equals(frame.data)


def test_range_attributes(featurelog):
    frame = FeatureMapping(featurelog, device='68:9E:19:07:DE:C3')
    assert len(frame) == 3
    frame = FeatureMapping(featurelog, device=['68:9E:19:07:DE:C3'])
    assert len(frame) == 3
    frame = FeatureMapping(featurelog, timestamp=('2016-9-14 10:03', None))
    assert len(frame) == 5
    frame = FeatureMapping(featurelog, ix_range=('2016-9-14 10:03', None))
    assert len(frame) == 5
    frame = FeatureMappingMK(featurelog, Feature_Value_0=0)
    assert len(frame) == 0
    with pytest.raises(frm.FrameError):
        frame = FeatureMapping(featurelog, Feature_Value_0=0)
    with pytest.raises(ValueError):
        frame = FeatureMapping(featurelog, ix_range=('2016-9-14 10:03', None),
                               timestamp=('2016-9-14 10:03', None))

if __name__ == '__main__':
    pytest.main([__file__])
