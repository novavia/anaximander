#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides an interface from the data package to Google BigQuery.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from google.cloud import bigquery as bq
from google.cloud.bigquery.client import Client
from google.cloud.bigquery.dataset import Dataset

from ..utilities import functions as fun, nxattr, xprops
from ..meta import prototype, metacharacter
from .fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant, Scalar, Timestamp, Duration, Period
from .schema import Schema
from .table import DataTable, DataQuery

__all__ = ['BigQueryDataTable', 'BigQueryQuery', 'Dataset', 'Client']

# =============================================================================
# Field mapping
# =============================================================================


# Mapping from base marshmallow field types to BigQuery field types
_field_map = {Field: 'STRING',
              Raw: 'STRING',
              Nested: 'RECORD',
              Dict: NotImplemented,
              List: NotImplemented,
              String: 'STRING',
              UUID: 'STRING',
              Number: 'FLOAT',
              Integer: 'INTEGER',
              Decimal: 'FLOAT',
              Boolean: 'BOOLEAN',
              FormattedString: 'STRING',
              Float: 'FLOAT',
              DateTime: 'TIMESTAMP',
              LocalDateTime: 'TIMESTAMP',
              Time: 'TIME',
              Date: 'DATE',
              TimeDelta: 'INTEGER',
              Url: 'STRING',
              URL: 'STRING',
              Email: 'STRING',
              Method: NotImplemented,
              Function: NotImplemented,
              Str: 'STRING',
              Bool: 'BOOLEAN',
              Int: 'INTEGER',
              Constant: NotImplemented,
              Timestamp:  'TIMESTAMP',
              Duration: 'INTEGER',
              Period: 'TIMESTAMP'
              }


# Mapping from numpy dtype kinds to BiqQuery field types
_dtype_kind_map = {'b': 'BOOLEAN',
                   'i': 'INTEGER',
                   'u': 'INTEGER',
                   'f': 'FLOAT',
                   'c': NotImplemented,  # complex floating-point
                   'm': 'INTEGER',
                   'M': 'TIMESTAMP',
                   'O': NotImplemented,  # Object
                   'S': 'STRING',  # (byte-)string
                   'U': 'STRING',  # Unicode
                   'V': NotImplemented,  # void
                   }


def _field_type(field):
    """Returns a BiqQuery field type from a Schema field or NotImpemented."""
    ftype = type(field)
    if issubclass(ftype, Scalar):
        return _dtype_kind_map.get(field.datatype.dtype.kind, NotImplemented)
    for ft in ftype.__mro__:
        try:
            return _field_map[ft]
        except KeyError:
            pass
    raise TypeError("Non-field type {0} passed to dtype.".format(field))


def bqfield(field):
    """Returns a BigQuery schema field from a Field instance.

    Args:
        field (obj): a Field instance.

    Returns:
        a BigQuery SchemaField instance.
    """
    name = field.name
    if name is None:
        msg = "Field names must be set in their metadata attributes " + \
            "in order to be usable in BigQuery schemas."
        raise ValueError(msg)
    field_type = _field_type(field)
    if field_type is NotImplemented:
        msg = "Cannot use field of type {0} in BigQuery schema."
        raise ValueError(msg.format(type(field).__name__))
    mode = 'REQUIRED' if field.required else 'NULLABLE'
    description = field.description
    try:
        # Serializes a nested field
        fields = bqschema(field.nested)
    except AttributeError:
        fields = None
    return bq.SchemaField(name, field_type, mode,
                          description=description, fields=fields)

# =============================================================================
# Schema specification and table creation
# =============================================================================


def bqschema(schema):
    """Makes a BiqQuery schema from a Schema instance.

    Args:
        schema (obj): a Schema instance.

    Returns:
        list of SchemaField instances.
    """
    try:
        return [bqfield(f) for f in schema.fields.values()]
    except ValueError:
        raise ValueError("Schema cannot be serialized.")

# =============================================================================
# Table implementation
# =============================================================================


@prototype
@nxattr.s
class BigQueryDataTable(DataTable):
    schema = metacharacter(validate=fun.subcheck(Schema))
    dataset = nxattr.ib(validator=nxattr.validators.instance_of(Dataset))
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))

    @xprops.cachedproperty
    def table(self):
        """Caches an instance of table in the bigquery API."""
        table = self.dataset.table(self.name, bqschema(self.schema))
        table.friendly_name = self.name
        return table

    @property
    def exists(self):
        return self.table.exists()

    @property
    def table_id(self):
        self.table.reload()
        return self.table.table_id

    @property
    def client(self):
        return self.dataset._client

    def __create__(self):
        return self.table.create()

    def __remove__(self):
        return self.table.delete()

    def __query__(self, *fields, **kwargs):
        return BigQueryQuery(self, *fields, **kwargs)

    def __insert__(self, record, **kwargs):
        row = self.schema.pythonize(record, mapping=False)
        self.table.insert_data([row], **kwargs)

    def __append__(self, frame, **kwargs):
        pdrows = frame.data.values
        rows = [self.schema.pythonize(tuple(r), mapping=False) for r in pdrows]
        self.table.insert_data(rows, **kwargs)


class BigQueryQuery(DataQuery):
    __sql__ = True

    def _sql_generator(self):
        """A basic SQL generator to handle simple queries."""
        select_ = "SELECT " + ", ".join(self.fields)
        from_ = "FROM [{}]".format(self.table.table_id)
        where_ = " AND ".join(v.sql(k) for k, v in self.quargs.items())
        if where_:
            where_ = "WHERE " + where_
        seqkey = self.table.schema.seqkey
        if seqkey:
            order_ = "ORDER BY " + seqkey
        else:
            order_ = ""
        parts = [select_, from_, where_, order_]
        return " ".join(parts)

    def __fetch__(self, limit=None, timeout=10000, pagecount=1000):
        """Fetch primitive.

        Params:
            limit: an optional limit on the total number of rows.
            timeout: timeout specification per BigQuery API.
            pagecount: max. number of results per page per BigQuery API.
        """
        results = self.table.client.run_sync_query(self.sql)
        limit = fun.get(limit, float('inf'))
        results.timeout_ms = timeout
        results.max_results = pagecount
        results.run()
        rowcount = min(results.total_rows, limit)
        print("Retrieved {0} rows from server.".format(rowcount))
        pg_token = None
        counter = 0
        while True:
            rows, _, pg_token = results.fetch_data(page_token=pg_token)
            for row in rows:
                yield row
                counter += 1
                if counter >= limit:
                    break
            print("Downloaded {} rows out of {}.".format(counter, rowcount))
            if not pg_token:
                break
