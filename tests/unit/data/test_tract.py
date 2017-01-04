#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for tract.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import datetime as dt
from unittest import TestCase

import pytest

from anaximander.data import fields, tract, record as rec, schema as sch

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def schemas():

    class UserSchema(sch.Schema):
        email = fields.Email(key=True)
        name = fields.String()

    class PurchaseSchema(sch.Schema):
        user = fields.Nested(UserSchema, key=True)
        timestamp = fields.DateTime(key=True, sequential=True,
                                    default=dt.datetime.utcnow)
        item = fields.String(key=True, default='miscellaneous')

    return UserSchema, PurchaseSchema


class TestTract(TestCase):

    def setUp(self):
        self.UserSchema, self.PurchaseSchema = schemas()

    def test_init(self):
        global User
        tract.Tract(self.UserSchema)
        assert 'User' in globals()
        assert User.name == 'User'
        assert User.Schema is self.UserSchema
        assert self.UserSchema.tract is User
        assert issubclass(User.Record, rec.Record)
        assert User.Record.__name__ == 'UserRecord'
        assert User.Record.schema is self.UserSchema

    def test_init_with_name(self):
        global NxUser
        tract.Tract(self.UserSchema, name='NxUser')
        assert 'NxUser' in globals()
        assert NxUser.name == 'NxUser'
        assert NxUser.Schema is self.UserSchema
        assert self.UserSchema.tract is NxUser

    def test_init_with_wrong_naming(self):

        class NxUserModel(self.UserSchema):
            pass

        global NxUser
        with pytest.raises(tract.TractDefinitionError):
            tract.Tract(NxUserModel)
        tract.Tract(NxUserModel, 'NxUser')
        assert NxUserModel.tract is NxUser

    def test_make_record_class(self):
        global User
        tract.Tract(self.UserSchema)
        assert hasattr(User.Record, 'email')
        assert hasattr(User.Record, 'name')


def test_decorator(schemas):
    UserSchema, PurchaseSchema = schemas
    global User
    tract.tract(UserSchema)
    assert 'User' in globals()
    assert User.name == 'User'
    assert User.Schema is UserSchema
    assert UserSchema.tract is User


def test_record_integration(schemas):
    UserSchema, PurchaseSchema = schemas
    global User, Purchase
    tract.tract(UserSchema)
    tract.tract(PurchaseSchema)

    # Load from schema
    user_data = {'name': 'Joe', 'email': 'joe@bar.com'}
    user = UserSchema().load(user_data).data
    assert type(user) == User.Record
    assert type(user).__name__ == 'UserRecord'

    # Load from record class
    purchase_data = {'user': user_data}
    purchase = Purchase.Record.load(purchase_data)
    assert type(purchase) == Purchase.Record
    assert type(purchase).__name__ == 'PurchaseRecord'

    # Verifies correctness of default field values
    assert purchase.item == 'miscellaneous'
    assert dt.datetime.utcnow() - purchase.timestamp < dt.timedelta(1)

    # Verifies that ValidationError is raised on bad schema
    with pytest.raises(sch.ValidationError):
        UserSchema().load({})

    # Verifies direct record instantiation
    assert purchase == Purchase.Record(**purchase.as_dict(recurse=False))
    purchase_data['timestamp'] = None
    assert Purchase.Record(**purchase_data).timestamp is None
    with pytest.raises(sch.ValidationError):
        Purchase.Record(validate=True, **purchase_data)


def test_nesting(schemas):
    UserSchema, PurchaseSchema = schemas
    global User, Purchase
    tract.tract(UserSchema)
    tract.tract(PurchaseSchema)

    user_data = {'name': 'Joe', 'email': 'joe@bar.com'}
    purchase_data = {'user': user_data,
                     'timestamp': '2016-12-14T17:14:48+00:00',
                     'item': 'hammer'}
    user = User.Record.load(user_data)
    purchase = Purchase.Record.load(purchase_data)
    assert purchase.dump() == purchase_data
    assert purchase.as_dict()['user'] == user.as_dict()


def test_schema_subclassing(schemas):
    """Tests subclassing of Schemas into Tracts."""
    UserSchema, PurchaseSchema = schemas
    global User, Purchase
    tract.tract(UserSchema)
    tract.tract(PurchaseSchema)

    @tract.tract
    class PowerUserSchema(UserSchema):
        power = fields.Bool(default=True)
    global PowerUser
    user_data = {'name': 'Joe', 'email': 'joe@bar.com', 'power': 'True'}
    user = PowerUser.Record(user_data)
    assert isinstance(user, User.Record)


def test_record_subclassing(schemas):
    """Tests subclassing a Record class."""
    UserSchema, PurchaseSchema = schemas
    global User, Purchase
    tract.tract(UserSchema)
    tract.tract(PurchaseSchema)

    class UserRecord(User.Record):
        @property
        def greetings(self):
            return "Hello, {0}.".format(self.name)

    user_data = {'name': 'Joe', 'email': 'joe@bar.com'}
    user = User.Record.load(user_data)
    assert isinstance(user, UserRecord)
    assert user.greetings == "Hello, Joe."

if __name__ == '__main__':
    pytest.main([__file__])
