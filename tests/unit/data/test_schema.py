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

import marshmallow as msh
import pytest

from anaximander.data import schema as sch

# =============================================================================
# Test Cases
# =============================================================================


def test_field_properties():

    class MyBaseSchema(sch.Schema):
        x = sch.Int(key=True)

    class MySchema(MyBaseSchema):
        y = sch.Bool(key=True)
        z = sch.DateTime(serial=True)

    x, y, z = MySchema.fields.values()
    assert MyBaseSchema.x == MySchema.x == x
    assert x.name is 'x'
    assert x.key is True
    assert y.key is True
    assert z.serial is True
    assert x.serial is False
    assert MyBaseSchema.fields == OrderedDict([('x', x)])
    assert MySchema.fields == OrderedDict([('x', x), ('y', y), ('z', z)])
    assert MySchema.keys == OrderedDict([('x', x), ('y', y)])
    assert MySchema.own_fields == OrderedDict([('y', y), ('z', z)])


def test_reserved_names():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            extra = sch.Int()


def test_field_type():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            x = msh.fields.Dict()


def test_field_missing():
    """Tests that specifying missing raises an error."""
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            x = sch.Int(missing=0)


def test_field_sequence():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            name = sch.Str()
            key = sch.Str(key=True)


if __name__ == '__main__':
    pytest.main([__file__])
