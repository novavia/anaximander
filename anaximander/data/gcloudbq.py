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
from .schema import Schema
from .frame import DataLoader, DataDumper
from .tract import DataTract, DataChannel

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


class _QBQDataQuery(DataLoader):
    """A primitive for concrete Query classes."""

    @property
    def table(self):
        return self.store

    def __load__(self):
        project = self.table.project
        return pdgbq.read_gbq(self.sql, project, **self.kwargs)


class GBQRawDataQuery(_QBQDataQuery):
    """A query that is instantiated with a raw SQl string.

    Args:
        schema: A Schema subtype.
        table: A gcloud.bigquery.table.Table instance.
        sql: A complete sql query statement.
        kwargs: keyword arguments that are passed to pandas.io.gbq.read_gbq.
    """

    def __init__(self, schema, table, sql, **kwargs):
        super().__init__(schema, table, **kwargs)
        self.sql = sql


@prototype
class GBQDataQuery(_QBQDataQuery):
    """A query with a simplified interface for simple statements.

    Args:
        table: A gcloud.bigquery.table.Table instance.
        kwargs: see generate_sql for admissible arguments.
    """
    schema = MetaCharacter(validate=fun.subcheck(Schema))
    # Admissible query arguments, which populate instance kwargs
    __quargs__ = ('columns', 'exclude', 'where', 'order_by', 'limit')
    # Default limit on the number of rows queried.
    __limit__ = 1000

    def __init__(self, table, **kwargs):
        self.store = table
        self.args = tuple()
        self.quargs = dict.fromkeys(self.__quargs__)
        self.quargs['limit'] = self.__limit__
        for k in kwargs:
            try:
                self.quargs[k] = kwargs.pop(k)
            except KeyError:
                continue
        self.kwargs = kwargs

    @property
    def sql(self):
        return self.generate_sql(self.store, **self.quargs)

    @classmethod
    def generate_sql(cls, table, columns=None, exclude=None, where=None,
                     order_by=None, limit=None):
        """A basic SQL generator to handle simple queries.

        More complex queries can resort to GBQRawDataQuery, which
        accepts raw SQL.

        Args:
            columns: an iterable of column names as strings. Defaults to
                None, which is equivalent to SQL (*).
            exclude: an iterable of column names as strings. These will be
                excluded from the results.
            where: a string containing a valid SQL where statement.
            order_by: a column name or iterable thereof used to order the data.
            limit (int): a limit to the number of rows to be returned.

        Returns:
            an SQL statement as a string.

        Raises:
            QueryException if both columns and exclude are specified, or if
            the arguments don't allow forming a valid SQL statement.
        """
        if exclude is not None:
            if columns is not None:
                msg = "Specify columns or exclude but not both."
                raise QueryException(msg)
            columns = set(cls.schema.fields) - set(exclude)
        if columns is not None:
            select_ = "SELECT " + ", ".join(columns)
        else:
            select_ = "SELECT *"
        from_ = "FROM [{}]".format(table.table_id)
        where_ = where
        if order_by is None:
            order_ = None
        elif isinstance(order_by, str):
            order_ = "ORDER BY " + order_by
        else:
            order_ = "ORDER BY " + ", ".join(order_by)
        limit_ = "LIMIT {}".format(limit) if limit is not None else None
        parts = [select_, from_, where_, order_, limit_]
        return " ".join(p for p in parts if p is not None)


class BigQueryChannel(DataChannel):
    __loader__ = GBQDataQuery

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
                assumes that the dataset follows the DataTract's table
                naming convention, but this can be overriden by
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
        return GBQRawDataQuery(self.schema, self.table, sql, **kwargs)

    def query(self, *args, **kwargs):
        return GBQDataQuery[self.schema](self.table, **kwargs)
