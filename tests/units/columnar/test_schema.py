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

from anaximander2.fields import Integer, Text
from anaximander2.columnar import schema as sch

# =============================================================================
# Tests
# =============================================================================


class MyFieldMap(sch.FieldMap):
    x = Integer()
    y = Text()


def test_field_map():
    f = MyFieldMap()
    assert isinstance(f['x'], Integer)
    assert isinstance(f['y'], Text)


class MyIndex(sch.SchemaIndex):
    id = Integer(sequencer=True)

    def rowkey(self, **record):
        return str(record['id'])

    def rowloc(self, key):
        return (int(key),)


def test_index():
    my_index = MyIndex()
    assert my_index.id.index
    assert my_index.rowkey(id=3) == '3'
    assert my_index.rowloc('3') == (3,)
    assert my_index.sequencer == my_index.id
    assert my_index.keys == []


class MySchema(sch.Schema):
    id = Integer(index=True)
    x = Text()

    def rowkey(self, **record):
        return str(record['id'])

    def rowloc(self, key):
        return (int(key),)


def test_schema():
    my_schema = MySchema()
    assert list(my_schema) == ['id', 'x']
    assert list(my_schema.index) == ['id']
    assert list(my_schema.columns) == ['x']
    assert my_schema['id'] is my_schema.index['id']
    assert my_schema['x'] is my_schema.columns['x']


class MyInheritedSchema(MySchema):
    y = Text()


def test_inherited_schema():
    inherited_schema = MyInheritedSchema()
    assert isinstance(inherited_schema, MySchema)
    assert list(inherited_schema) == ['id', 'x', 'y']
    assert list(inherited_schema.index) == ['id']
    assert list(inherited_schema.columns) == ['x', 'y']


class MyAltSchema(sch.Schema, index=MyIndex):
    x = Text()


def test_alt_schema():
    my_alt_schema = MyAltSchema()
    assert list(my_alt_schema) == ['id', 'x']
    assert list(my_alt_schema.index) == ['id']
    assert list(my_alt_schema.columns) == ['x']
    assert my_alt_schema['id'] is my_alt_schema.index['id']
    assert my_alt_schema['x'] is my_alt_schema.columns['x']


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
    a = Integer(index=True)
    b = Text()


def test_out_of_order_schema():
    schema = OutOfOrderSchema()
    assert list(schema) == ['id', 'a', 'x', 'b']
    assert list(schema.index) == ['id', 'a']
    assert list(schema.columns) == ['x', 'b']


def test_errors():
    with pytest.raises(sch.SchemaError):
        class FieldMap(sch.FieldMap):
            pass
        FieldMap()
    with pytest.raises(sch.SchemaError):
        class Schema(sch.Schema, index=MyIndex):
            x = Integer(index=True)
            y = Text()
    with pytest.raises(sch.SchemaError):
        class Index(sch.SchemaIndex):
            a = Integer(index=True, sequencer=True)
            b = Text(sequencer=True)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
