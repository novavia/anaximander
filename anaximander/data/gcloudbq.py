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

from gcloud import bigquery as bq
from pandas.io import gbq as pdgbq

from ..utilities import functions as fun
from ..meta.nxtype import prototype
from ..meta.metadescriptors import MetaCharacter
from . import channel as chn
from .tract import DataTract

# =============================================================================
# Custom exceptions
# =============================================================================


class BigQueryException(Exception):
    """Exception related to BigQuery administration."""
    pass


class QueryException(Exception):
    """Exception related to querying."""
    pass


class _QueryResultsPatch:
    """Adds a row generator for convenience."""

    def fetch_all(self):
        """A row generator."""
        if not self.complete:
            self.run()
        pg_token = None
        while True:
            rows, _, pg_token = self.fetch_data(page_token=pg_token)
            for row in rows:
                yield row
            if not pg_token:
                break

    def all(self):
        """Returns all results as a list."""
        return list(self.fetch_all())

    def first(self):
        """Returns the first row if it exists or raise QueryException."""
        try:
            return next(self.fetch_all())
        except StopIteration:
            msg = "Query returns no results."
            raise QueryException(msg)

fun.monkeypatch(bq.query.QueryResults, _QueryResultsPatch)

# =============================================================================
# Field mapping
# =============================================================================


_field_map = {'Field': 'STRING',
              'Raw': 'STRING',
              'Nested': 'RECORD',
              'Dict': NotImplemented,
              'List': NotImplemented,
              'String': 'STRING',
              'UUID': 'STRING',
              'Number': 'FLOAT',
              'Integer': 'INTEGER',
              'Decimal': 'FLOAT',
              'Boolean': 'BOOLEAN',
              'FormattedString': 'STRING',
              'Float': 'FLOAT',
              'DateTime': 'TIMESTAMP',
              'LocalDateTime': 'TIMESTAMP',
              'Time': 'TIME',
              'Date': 'DATE',
              'TimeDelta': 'INTEGER',
              'Url': 'STRING',
              'URL': 'STRING',
              'Email': 'STRING',
              'Method': NotImplemented,
              'Function': NotImplemented,
              'Str': NotImplemented,
              'Bool': NotImplemented,
              'Int': NotImplemented,
              'Constant': NotImplemented,
              }


def _field_type(field):
    """Returns a BiqQuery field type from a Schema field or NotImpemented."""
    return _field_map.get(type(field).__name__, NotImplemented)


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
# Channel, Loader and Dumper specializations
# =============================================================================


class BigQueryChannel(chn.DataChannel):

    @property
    def table(self):
        """Alias for the channel's store, which is always a table."""
        return self.store

    @classmethod
    def from_dataset(cls, dataset, tract, tbname=None):
        """Instantiates a channel from a BQ Dataset instance.

        Args:
            dataset: a BigQuery Dataset instance per official API.
            tract: a DataTract instance.
            tbname: an optional table name. By default, the method
                assumes that the tables in the dataset follow tract
                naming conventions, but this can be overriden by
                specifying the table name.

        Returns:
            a BigQueryChannel instance.
        """
        tbname = tbname or tract.tbname
        table = dataset.table(tbname)
        if not table.exists():
            msg = "Cannot create a channel to a non-existing table."
            raise BigQueryException(msg)
        table.reload()
        return cls(tract, table)

    def rawquery(self, sql, **kwargs):
        return GBQRawDataQuery(self, sql, **kwargs)

    def query(self, *args, **kwargs):
        return GBQDataQuery(self, *args, **kwargs)


class _QBQDataQuery(chn.DataLoader):
    """A primitive for concrete Query classes."""

    def __run__(self):
        project = self.channel.table.project
        return pdgbq.read_gbq(self.sql, project, **self.kwargs)


class GBQRawDataQuery(_QBQDataQuery):
    """A query that is instantiated with a raw SQl string.

    Args:
        channel: A BigQueryChannel instance.
        sql: A complete sql query statement.
        kwargs: keyword arguments that are passed to pandas.io.gbq.read_gbq.
    """

    def __init__(self, channel, sql, **kwargs):
        super().__init__(channel)
        self.sql = sql
        self.kwargs = kwargs


# TODO: define interface --work iteratively starting with simple statements.
@prototype
class GBQDataQuery(_QBQDataQuery):
    """A query with a simplified interface for simple statements.

    Args:
        channel: A BigQueryChannel instance.
        # TODO:
        rest is TBD.
    """
    tract = MetaCharacter(validate=fun.typecheck(DataTract))

    def __init__(self, channel, *args, **kwargs):
        super().__init__(channel)
        self.sql = self.generate_sql(*args, **kwargs)
        self.kwargs = kwargs
