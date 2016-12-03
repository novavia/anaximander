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
import datetime as dt

import pytest

from anaximander.data import schemas as sch

# =============================================================================
# Test Cases
# =============================================================================


def test_field_properties():

    class MyBaseSchema(sch.Schema):
        x = sch.fields.Int(key=True)

    class MySchema(MyBaseSchema):
        y = sch.fields.Bool(key=True)
        z = sch.fields.DateTime(serial=True)

    x, y, z = MySchema.fields.values()
    assert MyBaseSchema.x == MySchema.x == x
    assert x.name is 'x'
    assert x.key is True
    assert y.key is True
    assert z.serial is True
    assert x.serial is False
    assert MySchema.keys == OrderedDict([('x', x), ('y', y)])


def test_reserved_names():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            extra = sch.fields.Int()


def test_field_type():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            x = sch.fields.Dict()


def test_field_missing():
    """Tests that specifying missing raises an error."""
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            x = sch.fields.Int(missing=0)


def test_field_sequence():
    with pytest.raises(sch.SchemaError):
        class MySchema(sch.Schema):
            name = sch.fields.Str()
            key = sch.fields.Str(key=True)


def test_load():

    class UserSchema(sch.Schema):
        email = sch.fields.Email(key=True)
        name = sch.fields.String()

    class PurchaseSchema(sch.Schema):
        user = sch.fields.Nested(UserSchema, key=True)
        timestamp = sch.fields.DateTime(key=True, serial=True,
                                        default=dt.datetime.utcnow)
        item = sch.fields.String(key=True, default='miscellaneous')

    user_data = {'name': 'Joe', 'email': 'joe@bar.com'}
    user = UserSchema().load(user_data).data
    assert type(user) == UserSchema.record_class
    assert type(user).__name__ == 'User'
    purchase_data = {'user': user_data}
    purchase = PurchaseSchema().load(purchase_data).data
    assert type(purchase).__name__ == 'Purchase'
    assert purchase.item == 'miscellaneous'
    assert dt.datetime.utcnow() - purchase.timestamp < dt.timedelta(1)
    with pytest.raises(sch.ValidationError):
        UserSchema().load({})

if __name__ == '__main__':
    pytest.main([__file__])
