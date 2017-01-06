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

from collections import OrderedDict

import pytest

from anaximander.data import fields, schema as sch

# =============================================================================
# Test Cases
# =============================================================================


def test_field_properties():

    class MyBaseSchema(sch.Schema):
        x = fields.Int(key=True)

    class MySchema(MyBaseSchema):
        y = fields.Bool(key=True)
        z = fields.DateTime(sequential=True)

    x, y, z = MySchema.fields.values()
    assert MyBaseSchema.x == MySchema.x == x
    assert x.name is 'x'
    assert x.key is True
    assert y.key is True
    assert z.sequential is True
    assert x.sequential is False
    assert MyBaseSchema.fields == OrderedDict([('x', x)])
    assert MySchema.fields == OrderedDict([('x', x), ('y', y), ('z', z)])
    assert MySchema.keys == OrderedDict([('x', x), ('y', y)])
    assert MySchema.own_fields == OrderedDict([('y', y), ('z', z)])


def test_reserved_names():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            extra = fields.Int()


def test_field_missing():
    """Tests that specifying missing raises an error."""
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            x = fields.Int(missing=0)


def test_field_sequence():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            name = fields.Str()
            key = fields.Str(key=True)


if __name__ == '__main__':
    pytest.main([__file__])
