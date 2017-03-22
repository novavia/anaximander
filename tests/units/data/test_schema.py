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

from collections import OrderedDict, Mapping
import datetime as dt

import pandas as pd
import pytest

from anaximander.utilities.nxtime import datetime, UTC
from anaximander.data import fields, schema as sch, quantities as qnt, \
    NxScalar

# =============================================================================
# Test Cases
# =============================================================================


def test_field_properties():

    class MyBaseSchema(sch.Schema):
        x = fields.Int(key=True)

    class MySchema(MyBaseSchema):
        y = fields.DateTime(key=True, sequential=True)
        z = fields.Bool()

    x, y, z = MySchema.fields.values()
    assert MyBaseSchema.x == MySchema.x == x
    assert x.name is 'x'
    assert x.key is True
    assert y.key is True
    assert y.sequential is True
    assert x.sequential is False
    assert MyBaseSchema.fields == OrderedDict([('x', x)])
    assert MySchema.fields == OrderedDict([('x', x), ('y', y), ('z', z)])
    assert MySchema.keys == OrderedDict([('x', x), ('y', y)])
    assert MySchema.nskeys == OrderedDict([('x', x)])
    assert MySchema.seqkey == 'y'
    assert MySchema.own_fields == OrderedDict([('y', y), ('z', z)])


def _rdict(dict_):
    """Makes a dict of mapping, recursively."""
    mapping = {}
    for k, v in dict_.items():
        if isinstance(v, Mapping):
            mapping[k] = _rdict(v)
        else:
            mapping[k] = v
    return mapping


def _rtuple(dict_):
    """Makes a tuple of a dict, recursively."""
    collector = []
    for v in dict_.values():
        if isinstance(v, Mapping):
            collector.append(_rtuple(v))
        else:
            collector.append(v)
    return tuple(collector)


def test_pythonize():

    length = qnt.Quantity('length', 'm')

    class Length(NxScalar):
        quantity = length

    class Person(sch.Schema):
        name = fields.Str(key=True)
        height = fields.Scalar(Length)

    class Order(sch.Schema):
        person = fields.Nested(Person, key=True)
        time = fields.Timestamp(key=True)
        thing = fields.Str()

    _serialized = OrderedDict([('person', OrderedDict([('name', 'Joe'),
                                                      ('height', 1.8)])),
                               ('time', '2017-02-17 15:00:00+00:00'),
                               ('thing', 'hammer')])
    _unserialized = OrderedDict([('person',
                                  OrderedDict([('name', 'Joe'),
                                               ('height', Length(1.8))])),
                                 ('time', datetime('2017-02-17 15:00:00')),
                                 ('thing', 'hammer')])
    _pythonized = OrderedDict([('person', OrderedDict([('name', 'Joe'),
                                                       ('height', 1.8)])),
                               ('time',
                                 dt.datetime(2017, 2, 17, 15, tzinfo=UTC)),
                               ('thing', 'hammer')])
    serialized = _rdict(_serialized)
    unserialized = _rdict(_unserialized)
    pythonized = _rdict(_pythonized)
    unserialized_t = _rtuple(_unserialized)
    pythonized_t = _rtuple(_pythonized)
    assert Order().load(serialized).data == unserialized
    assert Order.pythonize(unserialized) == pythonized
    assert Order.depythonize(pythonized) == unserialized
    assert Order().pythonize(unserialized_t) == pythonized
    assert Order().depythonize(pythonized_t) == unserialized
    assert Order().pythonize(unserialized, mapping=False) == pythonized_t
    assert Order().depythonize(pythonized, mapping=False) == unserialized_t
    assert Order().pythonize(unserialized_t, mapping=False) == pythonized_t
    assert Order().depythonize(pythonized_t, mapping=False) == unserialized_t


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


def test_unique_sequential_key():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            a = fields.Int(key=True, sequential=True)
            b = fields.Int(key=True, sequential=True)


if __name__ == '__main__':
    pytest.main([__file__])
