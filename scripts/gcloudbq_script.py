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

from contextlib import contextmanager

from gcloud import bigquery as bq

from anaximander.data import fields, schema as sch, gcloudbq as gbq


PROJECT_ID = 'infinite-uptime-1232'
BQ = bq.Client(PROJECT_ID)
DATASET_ID = 'InterfaceTesting'

# =============================================================================
# Test Cases
# =============================================================================


def cleanup(dataset):
    """Cleans up the supplied BigQuery dataset."""
    for table in dataset.list_tables()[0]:
        table.delete()
    dataset.delete()


@contextmanager
def session():
    dataset = BQ.dataset(DATASET_ID)
    if dataset.exists():
        cleanup(dataset)
    dataset.create()
    yield dataset  # provide the fixture value
#    cleanup(dataset)


class UserSchema(sch.Schema):
    name = fields.String()
    email = fields.Email(key=True)


class PurchaseSchema(sch.Schema):
    user = fields.Nested(UserSchema, key=True)
    timestamp = fields.DateTime(key=True)
    item = fields.String(key=True)


class BasketSchema(sch.Schema):
    user = fields.Nested(UserSchema, key=True)
    items = fields.List(fields.String())


def test_bqfield():
    user = gbq.bqfield(PurchaseSchema.user)
    timestamp = gbq.bqfield(PurchaseSchema.timestamp)
    item = gbq.bqfield(PurchaseSchema.item)
    assert isinstance(user, gbq.bq.SchemaField)
    assert isinstance(timestamp, gbq.bq.SchemaField)
    assert isinstance(item, gbq.bq.SchemaField)
    assert user.field_type is 'RECORD'
    assert timestamp.field_type is 'TIMESTAMP'
    assert item.field_type is 'STRING'
    assert [f.name for f in user.fields] == ['name', 'email']


def test_create_table(dataset):
    name, schema = 'Users', gbq.bqschema(UserSchema)
    table = dataset.table(name, schema)
    table.create()
    assert table.exists()

if __name__ == '__main__':
    with session() as dataset:
        user_table = dataset.table('Users',
                                   gbq.bqschema(UserSchema))
        purchase_table = dataset.table('Purchases',
                                       gbq.bqschema(PurchaseSchema))
        user_table.create()
        purchase_table.create()
