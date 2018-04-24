#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for schema.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from collections import OrderedDict

import pytest

from anaximander3.data.nxcolumns import Integer, String, Text, DateTime, \
    Float, EventLabel
from anaximander3.data import nxschema as sch

# =============================================================================
# Tests
# =============================================================================


class MyColumnMap(sch.ColumnMap):
    x = Integer()
    y = Text()


def test_column_map():
    f = MyColumnMap()
    assert isinstance(f['x'], Integer)
    assert isinstance(f['y'], Text)


class MyIndex(sch.SchemaIndex):
    id = Integer(index='sequential')

    def __rowkey__(self, idx):
        return str(idx[0])

    def __rowidx__(self, key):
        return (int(key),)


def test_index():
    my_index = MyIndex()
    assert my_index.id.index
    assert my_index.rowkey((3,)) == '3'
    assert my_index.rowidx('3') == (3,)
    assert my_index.sequencer == 'id'
    assert my_index.identifier is None


class MySchema(sch.Schema):
    id = Integer(index='nominal')
    x = Text()

    def __rowkey__(self, idx):
        return str(idx[0])

    def __rowidx__(self, key):
        return (int(key),)


def test_schema():
    my_schema = MySchema()
    assert list(my_schema) == ['id', 'x']
    assert list(my_schema.index) == ['id']
    assert list(my_schema.payload) == ['x']
    assert my_schema['id'] is my_schema.index['id']
    assert my_schema['x'] is my_schema.payload['x']


def test_json():
    my_schema = MySchema()
    assert MySchema.json_loads(my_schema.json_dumps()) == my_schema


class MyInheritedSchema(MySchema):
    y = Text()


def test_inherited_schema():
    inherited_schema = MyInheritedSchema()
    assert isinstance(inherited_schema, MySchema)
    assert list(inherited_schema) == ['id', 'x', 'y']
    assert list(inherited_schema.index) == ['id']
    assert list(inherited_schema.payload) == ['x', 'y']


class MyAltSchema(sch.Schema, index=MyIndex):
    x = Text()


def test_alt_schema():
    my_alt_schema = MyAltSchema()
    assert list(my_alt_schema) == ['id', 'x']
    assert list(my_alt_schema.index) == ['id']
    assert list(my_alt_schema.payload) == ['x']
    assert my_alt_schema['id'] is my_alt_schema.index['id']
    assert my_alt_schema['x'] is my_alt_schema.payload['x']


class MyIndexedColumn(sch.IndexedColumn, index=MyIndex):
    x = Text()


def test_indexed_column():
    ixcol = MyIndexedColumn()
    assert list(ixcol) == ['id', 'x']
    assert list(ixcol.index) == ['id']
    assert ixcol['id'] is ixcol.index['id']
    assert ixcol['x'] is ixcol.column


class MyAltIndexedColumn(sch.IndexedColumn, index=MyIndex, column=MySchema.x):
    pass


def test_alt_indexed_column():
    ixcol = MyAltIndexedColumn()
    assert list(ixcol) == ['id', 'x']
    assert list(ixcol.index) == ['id']
    assert ixcol['id'] is ixcol.index['id']
    assert ixcol['x'] is ixcol.column


class MyFamily(sch.ColumnFamily):
    x = Text()
    y = Text()


class MyMultiSchema(sch.MultiSchema, index=MyIndex):
    a = MyFamily()
    b = MyFamily()


def test_multi_schema():
    multi_schema = MyMultiSchema()
    assert list(multi_schema) == ['id', 'a', 'b']
    assert list(multi_schema.index) == ['id']
    assert list(multi_schema.families) == ['a', 'b']
    assert multi_schema['id'] is multi_schema.index['id']
    assert multi_schema['a'] is multi_schema.families['a']
    assert multi_schema['a'] is not MyMultiSchema.__families__.\
        __nxcolumns__['a']
    assert multi_schema['a']['x'] == multi_schema['b']['x']


def test_errors():
    with pytest.raises(sch.SchemaError):
        class ColMap(sch.ColumnMap):
            pass
        ColMap()
    with pytest.raises(sch.SchemaError):
        class Schema(sch.Schema, index=MyIndex):
            x = Integer(index='nominal')
            y = Text()
    with pytest.raises(sch.SchemaError):
        class Index(sch.SchemaIndex):
            a = Integer(index='sequential')
            b = Text(index='sequential')


class RedefinedSampleSchema(sch.SampleLogsSchema):
    id = String(index='nominal')
    datetime = DateTime(tz='utc', index='sequential')
    temperature = Float()


class SoftEventSchema(sch.SoftEventLogsSchema):
    label = EventLabel(['a', 'b', 'c'])


def test_redefined_schemas():
    schema = RedefinedSampleSchema()
    assert isinstance(schema.index, sch.TimeSeriesIndex)
    schema = SoftEventSchema()
    assert isinstance(schema.index, sch.TimeSeriesIndex)
    assert list(schema) == ['id', 'datetime', 'end_time', 'label']


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
