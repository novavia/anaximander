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

import pytest

from anaximander.data import fields, schema as sch, record as rec

# =============================================================================
# Test Cases
# =============================================================================


class TestRecord(ut.TestCase):

    def setUp(self):

        class Schema(sch.Schema):
            title = fields.String(key=True)
            date = fields.Date(key=True)
            text = fields.String()

        self.schema = Schema()
        self.Record = rec.Record[Schema]
        self.data = {'title': 'ping',
                     'date': '2016-12-08',
                     'text': 'pong'}

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
                                    'date': dt.date(2016, 12, 8),
                                    'text': 'pong'}

    def test_as_tuple(self):
        record = self.schema.load(self.data).data
        assert record.as_tuple() == ('ping', dt.date(2016, 12, 8), 'pong')

    def test_validate(self):
        record = self.schema.load(self.data).data
        record.validate()
        del self.data['text']
        record = self.Record(**self.data)
        with pytest.raises(sch.ValidationError):
            record.validate()


if __name__ == '__main__':
    pytest.main([__file__])
