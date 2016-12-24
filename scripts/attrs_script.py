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

from anaximander.data import fields, schema as sch, tract as trc

# =============================================================================
# Schemas
# =============================================================================


@trc.tract
class UserSchema(sch.Schema):
    email = fields.Email(key=True)
    name = fields.String(default='')


@trc.tract
class PurchaseSchema(sch.Schema):
    user = fields.Nested(UserSchema, key=True)
    timestamp = fields.DateTime(key=True)
    item = fields.String(key=True)


def now():
    dtnow = dt.datetime.utcnow()
    return dtnow.strftime('%Y-%m-%d %H:%M:%S')


class PingSchema(sch.Schema):
    timestamp = fields.DateTime(default=dt.datetime.utcnow)


#    with pytest.raises(sch.ValidationError):
try:
    UserSchema().load({})
except Exception as e:
    assert type(e) == sch.ValidationError

try:
    class MySchema(sch.Schema):
        extra = fields.Int()
except Exception as e:
    assert type(e) == sch.SchemaError


@trc.tract
class UserSchema(sch.Schema):
    email = fields.Email(key=True)
    name = fields.String()


@trc.tract
class PowerUserSchema(UserSchema):
    power = fields.Bool(default=True)

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
    joe = User.Record('joe@gmail.com', 'Joe')
    purchase = Purchase.Record(joe, dt.datetime.utcnow(), 'hammer')
