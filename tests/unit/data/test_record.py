#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for schemas.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import datetime as dt
import unittest as ut

import attr
import marshmallow as msh
import pytest

from anaximander.data import schema as sch, record as rec

# =============================================================================
# Test Cases
# =============================================================================


class TestRecord(ut.TestCase):

    def setUp(self):

        class Schema(sch.Schema):
            title = sch.String(key=True)
            date = sch.Date(key=True)
            text = sch.String()

            @msh.post_load
            def record(self, data):
                return Record(**data)

        @attr.s
        class Record(rec.Record):
            title = attr.ib()
            date = attr.ib()
            text = attr.ib(default=None)

            @property
            def schema(self):
                """Returns the schema type associated with self."""
                return Schema

        self.schema = Schema()
        self.Record = Record
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
