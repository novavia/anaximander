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

from collections.abc import Mapping

from gcloud import bigquery as bq
from pandas.io import gbq as pdgbq

from ..utilities import functions as fun
from ..meta.nxtype import prototype
from ..meta.metadescriptors import MetaCharacter
from .fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant, Scalar
from .schema import Schema
from .frame import DataLoader, DataDumper, NxDataFrame
from .tract import DataChannel, DataDomain

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


def create_table(dataset, tract, tbname=None):
    """Creates a table from a DataTract in the specified dataset.

    Args:
        dataset: a BigQuery Dataset instance per official API.
        tract: a DataTract instance.
        tbname: an optional table name. By default, the method
            assumes that the dataset follows the DataTract's table
            naming convention, but this can be overriden by
            specifying the table name.

    Returns:
        The created table, as a BigQuery API instance.
    """
    tbname = tbname or tract.tbname
    schema = bqschema(tract.Schema)
    table = dataset.table(tbname, schema)
    table.friendly_name = tbname
    table.create()
    return table


def create_all(dataset, domain):
    """Creates all tables for a DataDomain in the specified dataset.

    Args:
        dataset: a BigQuery Dataset instance per official API.
        domain: a DataDomain instance or name as a string. Alternatively,
            a mapping from Tract to table name can be provided, in
            which case the mapping's values will be used in lieu of
            each Tract's name for table naming.

    Returns:
        The set of created tables.
    """
    if isinstance(domain, str):
        domain = DataDomain[domain]
    if isinstance(domain, Mapping):
        return set(create_table(dataset, t, n) for t, n in domain.items())
    else:
        return set(create_table(dataset, tract) for tract in domain)


def create_dataset(client, domain, name=None):
    """Creates a dataset from a DataDomain in the specified BQ client.

    Args:
        client: A BigQuery Client instance.
        domain: A DataDomain instance or name as a string.
        name: The name of the newly created dataset. Defaults to the
            domain's name.
    """
    if isinstance(domain, str):
        domain = DataDomain[domain]
    name = name or domain.name
    dataset = client.dataset(name)
    dataset.create()
    create_all(dataset, domain)
    return dataset

# =============================================================================
# Channel, Loader and Dumper specializations
# =============================================================================


class _QBQDataQuery(DataLoader):
    """A primitive for concrete Query classes."""

    @property
    def table(self):
        return self.store

    def __load__(self):
        try:
            self.table.reload()
        except:
            msg = "Invalid or non-existing table passed to query."
            raise BigQueryException(msg)
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
        for k in self.__quargs__:
            try:
                self.quargs[k] = kwargs.pop(k)
            except KeyError:
                continue
        self.kwargs = kwargs

    @property
    def sql(self):
        return self.generate_sql(self.store, **self.quargs)

    # TODO: make order_by be the sequential key, if any, by default.
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


class GBQDataAppend(DataDumper):
    """Writes from a NxDataFrame to a BigQuery table.

    Args:
        frame: an NxDataFrame instance
        table: a BigQuery table object from the client library.
    """

    def __init__(self, frame, table, **kwargs):
        super().__init__(frame, table, **kwargs)

    @property
    def table(self):
        return self.store

    @staticmethod
    def _to_rows(dataframe):
        """Returns BQ-compatible rows from a pandas dataframe."""
        records = dataframe.values
        # Conversion sequence
        cv_seq = []
        for dt in dataframe.dtypes:
            if dt.kind == 'M':
                cv_seq.append(lambda x: x.to_pydatetime())
            else:
                cv_seq.append(lambda x: x)

        def make_tuple(record):
            return tuple(cv(x) for x, cv in zip(record, cv_seq))

        return [make_tuple(r) for r in records]

    def __dump__(self):
        try:
            self.table.reload()
        except:
            msg = "Invalid or non-existing table passed to append statement."
            raise BigQueryException(msg)
        rows = self._to_rows(self.frame.data)
        return self.table.insert_data(rows)


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
        table.friendly_name = tbname
        if not table.exists():
            msg = "Cannot create a channel to a non-existing table."
            raise BigQueryException(msg)
        return cls(tract, table)

    def rawquery(self, sql, **kwargs):
        """Returns a raw query object from supplied sql statement.

        Args:
            sql: a sql statement as a string.
            **kwargs: keyword arguments passed to pandas' read_gbq.

        Returns:
            A QBQRawDataQuery object.
        """
        return GBQRawDataQuery(self.schema, self.table, sql, **kwargs)

    def query(self, **kwargs):
        """Returns a query object from supplied sql statement.

        Args:
            **kwargs: keyword arguments dispatched to GBQDataQuery's __init__
                method, then to pandas' read_gbq.

        Returns:
            A QBQDataQuery object.
        """
        return GBQDataQuery[self.schema](self.table, **kwargs)

    def append(self, frame, **kwargs):
        """Appends data in frame to self's table.

        Args:
            frame: an NxDataFrame instance, whose schema must match
                self's schema.
            **kwargs: kwargs passed to bq.table.Table.insert_data.

        Raises:
            TypeError if frame is not of the proper type.

        Returns:
            Return value from pandas.io.gpq.to_gbq.
        """
        if not isinstance(frame, NxDataFrame) or frame.schema != self.schema:
            msg = "Incorrect data frame type supplied to BigQuery channel."
            raise TypeError(msg)
        return GBQDataAppend(frame, self.table, **kwargs)()

    # TODO: improve to more direct insert once the type of timestamps in
    # records is sorted out.
    def insert(self, *records, **kwargs):
        """Insert one or more records to self's table.

        Args:
            *records: one or more records of class self.tract.Record.
            **kwargs: kwargs passed to bq.table.Table.insert_data.

        Returns:
            Return value from bq.table.Table.insert_data.
        """
        frame = self.tract.Frame.from_records(records)
        return self.append(frame, **kwargs)
