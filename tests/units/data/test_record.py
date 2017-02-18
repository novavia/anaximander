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

import datetime as dt
import unittest as ut

import pandas as pd
import pytest

from anaximander.data import fields, schema as sch, record as rec
from anaximander.data.data import NxFloat

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
        self.Record = rec.NxRecord[Schema]
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
                                    'date': pd.Timestamp('2016-12-08'),
                                    'value': NxFloat(3.5)}

    def test_as_pydict(self):
        record = self.schema.load(self.data).data
        assert record.as_pydict() == {'title': 'ping',
                                      'date': dt.datetime(2016, 12, 8),
                                      'value': 3.5}

    def test_as_tuple(self):
        record = self.schema.load(self.data).data
        assert record.as_tuple() == ('ping', pd.Timestamp('2016-12-08'),
                                     NxFloat(3.5))

    def test_as_pytuple(self):
        record = self.schema.load(self.data).data
        assert record.as_pytuple() == ('ping', dt.datetime(2016, 12, 8), 3.5)

    def test_from_pydict(self):
        record = self.schema.load(self.data).data
        d = {'title': 'ping', 'date': dt.datetime(2016, 12, 8), 'value': 3.5}
        assert self.Record.from_pydict(d) == record

    def test_from_pytuple(self):
        record = self.schema.load(self.data).data
        t = ('ping', dt.datetime(2016, 12, 8), 3.5)
        assert self.Record.from_pytuple(t) == record

    def test_data(self):
        record = self.schema.load(self.data).data
        series = pd.Series(['ping', dt.datetime(2016, 12, 8), 3.5],
                           index=['title', 'date', 'value'])
        assert series.equals(record.data)

    def test_validate(self):
        record = self.schema.load(self.data).data
        record.validate()
        record = self.Record(title='ping', date='pong')
        with pytest.raises(sch.ValidationError):
            record.validate()


if __name__ == '__main__':
    pytest.main([__file__])
