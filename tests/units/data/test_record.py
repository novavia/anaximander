#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for record.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import unittest as ut

import pandas as pd
import pytest

from anaximander.utilities.nxtime import datetime, pydatetime
from anaximander.data import fields, schema as sch, NxFloat, NxRecord, Sample

# =============================================================================
# Test Cases
# =============================================================================


class TestRecord(ut.TestCase):

    def setUp(self):

        class Schema(sch.Schema):
            title = fields.String(key=True)
            date = fields.Timestamp(key=True)
            value = fields.Scalar(NxFloat)

        self.schema = Schema()
        self.Record = NxRecord[Schema]
        self.data = {'title': 'ping',
                     'date': '2016-12-08',
                     'value': 3.5}

    def test_load(self):
        record = self.schema.load(self.data).data
        assert type(record) == self.Record

    def test_load_dump_load(self):
        record = self.schema.load(self.data).data
        assert type(record) == self.Record
        data = self.schema.dump(record).data
        assert record == self.schema.load(data).data

    def test_as_dict(self):
        record = self.schema.load(self.data).data
        assert record.as_dict() == {'title': 'ping',
                                    'date': datetime('2016-12-08'),
                                    'value': NxFloat(3.5)}

    def test_as_pydict(self):
        record = self.schema.load(self.data).data
        assert record.as_pydict() == {'title': 'ping',
                                      'date': pydatetime('2016-12-08'),
                                      'value': 3.5}

    def test_as_tuple(self):
        record = self.schema.load(self.data).data
        assert record.as_tuple() == ('ping', datetime('2016-12-08'),
                                     NxFloat(3.5))

    def test_as_pytuple(self):
        record = self.schema.load(self.data).data
        assert record.as_pytuple() == ('ping', pydatetime('2016-12-08'), 3.5)

    def test_from_pydict(self):
        record = self.schema.load(self.data).data
        d = {'title': 'ping', 'date': pydatetime('2016-12-08'), 'value': 3.5}
        assert self.Record.from_pydict(d) == record

    def test_from_pytuple(self):
        record = self.schema.load(self.data).data
        t = ('ping', pydatetime('2016-12-08'), 3.5)
        assert self.Record.from_pytuple(t) == record

    def test_data(self):
        record = self.schema.load(self.data).data
        series = pd.Series(['ping', pydatetime('2016-12-08'), 3.5],
                           index=['title', 'date', 'value'])
        assert series.equals(record.data)

    def test_validate(self):
        record = self.schema.load(self.data).data
        record.validate()
        record = self.Record(title='ping', date='pong')
        with pytest.raises(sch.ValidationError):
            record.validate()


def test_sample_record():

    class MySchema(sch.SampleLogSchema):
        value = fields.Float()

    MySample = NxRecord[MySchema]
    data = {'timestamp':'2017-3-7 16:12', 'value':'1.0'}
    sample = MySchema().load(data).data
    assert sample.timestamp == datetime('2017-3-7 16:12')
    assert isinstance(sample, MySample)
    assert issubclass(MySample, Sample)


if __name__ == '__main__':
    pytest.main([__file__])
