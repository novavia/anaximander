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

from anaximander.data import schema as sch

# =============================================================================
# Test Cases
# =============================================================================


def test_load():

    class UserSchema(sch.Schema):
        email = sch.fields.Email(key=True)
        name = sch.fields.String()

    class PurchaseSchema(sch.Schema):
        user = sch.fields.Nested(UserSchema, key=True)
        timestamp = sch.fields.DateTime(key=True, serial=True,
                                        default=dt.datetime.utcnow)
        item = sch.fields.String(key=True, default='miscellaneous')

    # Load from schema
    user_data = {'name': 'Joe', 'email': 'joe@bar.com'}
    user = UserSchema().load(user_data).data
    assert type(user) == UserSchema.record_class
    assert type(user).__name__ == 'User'

    # Load from record class
    purchase_data = {'user': user_data}
    purchase = PurchaseSchema.record_class.load(purchase_data)
    assert type(purchase) == PurchaseSchema.record_class
    assert type(purchase).__name__ == 'Purchase'

    # Verifies correctness of default field values
    assert purchase.item == 'miscellaneous'
    assert dt.datetime.utcnow() - purchase.timestamp < dt.timedelta(1)

    # Verifies that ValidationError is raised on bad schema
    with pytest.raises(sch.ValidationError):
        UserSchema().load({})

    # Verifies direct record instantiation
    Purchase = PurchaseSchema.record_class
    assert purchase == Purchase(**purchase.as_dict(recurse=False))
    purchase_data['timestamp'] = None
    assert Purchase(**purchase_data).timestamp is None
    with pytest.raises(sch.ValidationError):
        Purchase(validate=True, **purchase_data)

if __name__ == '__main__':
    pytest.main([__file__])
