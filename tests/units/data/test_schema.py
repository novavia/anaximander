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

import pytest

from anaximander2.data.columns import Integer, Text
from anaximander2.data import schema as sch

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

    def rowkey(self, **record):
        return str(record['id'])

    def rowidx(self, key):
        return (int(key),)


def test_index():
    my_index = MyIndex()
    assert my_index.id.index
    assert my_index.rowkey(id=3) == '3'
    assert my_index.rowidx('3') == (3,)
    assert my_index.sequencer == my_index.id
    assert my_index.identifiers == []


class MySchema(sch.Schema):
    id = Integer(index='nominal')
    x = Text()

    def rowkey(self, **record):
        return str(record['id'])

    def rowidx(self, key):
        return (int(key),)


def test_schema():
    my_schema = MySchema()
    assert list(my_schema) == ['id', 'x']
    assert list(my_schema.index) == ['id']
    assert list(my_schema.payload) == ['x']
    assert my_schema['id'] is my_schema.index['id']
    assert my_schema['x'] is my_schema.payload['x']


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
    assert multi_schema['a']['x'] != multi_schema['b']['x']


class OutOfOrderSchema(MySchema):
    a = Integer(index='nominal')
    b = Text()


def test_out_of_order_schema():
    schema = OutOfOrderSchema()
    assert list(schema) == ['id', 'a', 'x', 'b']
    assert list(schema.index) == ['id', 'a']
    assert list(schema.payload) == ['x', 'b']


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


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
