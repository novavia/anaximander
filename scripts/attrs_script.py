#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for BigQuery interface.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import datetime as dt
import re

import attr

from anaximander.data import schemas as sch
from anaximander.utilities import functions as fun

# =============================================================================
# Schemas
# =============================================================================


class UserSchema(sch.Schema):
    email = sch.fields.Email(key=True)
    name = sch.fields.String(default='')


class PurchaseSchema(sch.Schema):
    user = sch.fields.Nested(UserSchema, key=True)
    timestamp = sch.fields.DateTime(key=True)
    item = sch.fields.String(key=True)


def now():
    dtnow = dt.datetime.utcnow()
    return dtnow.strftime('%Y-%m-%d %H:%M:%S')


class PingSchema(sch.Schema):
    timestamp = sch.fields.DateTime(default=dt.datetime.utcnow)


#    with pytest.raises(sch.ValidationError):
try:
    UserSchema().load({})
except Exception as e:
    assert type(e) == sch.ValidationError

try:
    class MySchema(sch.Schema):
        extra = sch.fields.Int()
except Exception as e:
    assert type(e) == sch.SchemaError

#def attributes(schema):
#    """Extract a list of attribute specifications from a schema."""
#    attrs = {}
#
#    def default(field):
#        """Extracts the default instantiation value from a field."""
#        field_missing = field.missing
#        if field_missing is sch.missing:
#            return None
#        elif callable(field_missing):
#            def factory():
#                return field.deserialize(field_missing())
#            return attr.Factory(factory)
#        else:
#            return field_missing
#
#    for k, v in schema.fields.items():
#        if v.required:
#            attrs[k] = attr.ib()
#        else:
#            attrs[k] = attr.ib(default=default(v))
#    return attrs
#
#
#def record_class(schema, name=None):
#    """Makes a record class from schema."""
#    if name is None:
#        try:
#            name = re.match('.*(?=Schema)', schema.__name__).group(0)
#        except AttributeError:
#            msg = "The schema has a non-standard name, therefore a name " + \
#                "must be supplied for the Record class."
#            raise ValueError(msg)
#    return attr.make_class(name, attributes(schema))
#
#User = record_class(UserSchema)
#
#Ping = record_class(PingSchema)
#
#
#@attr.s
#class UserList:
#    users = attr.ib()

if __name__ == '__main__':
    pass
