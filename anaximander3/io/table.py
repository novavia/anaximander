#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the DataTable base class for data storage.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc
from collections import OrderedDict

from grpc._channel import _Rendezvous
import pandas as pd

from ..utilities import nxrange
from ..utilities.datastore import StorageResource
from ..data import datalogs as dtl, records as rcd


__all__ = ['DataTableException', 'DataTableAdminException',
           'DataTableReadException', 'DataTableWriteException',
           'EmptyQueryException', 'DataTable', 'DataQuery']

# =============================================================================
# Custom exceptions
# =============================================================================


class DataTableException(Exception):
    """Exception related to data tables."""
    pass


class DataTableAdminException(DataTableException):
    """Exception related to table administration."""
    pass


class DataTableReadException(DataTableException):
    """Exception related to reading data from a table."""
    pass


class DataTableWriteException(DataTableException):
    """Exception related to writing / updating data in a table."""
    pass

# =============================================================================
# Class declaration
# =============================================================================


class DataTable(StorageResource):
    """Abstract base class for data storage resources.

    Basically a light wrapper around table objects defined by various
    database APIs to unify basic commands.
    """

    def __init__(self, name, schema):
        self.name = name
        self.schema = schema

    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def __query__(self, *columns, **kwargs):
        pass

    def query(self, *columns, **kwargs):
        """Returns a Query object."""
        return self.__query__(*columns, **kwargs)

    @abc.abstractmethod
    def __insert__(self, record, **kwargs):
        pass

    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        for record in records:
            if not isinstance(record, rcd.Record):
                msg = f"Table insert requires a record, not a " + \
                      f"{type(record)} instance."
                raise TypeError(msg)
            if not isinstance(record.schema, type(self.schema)):
                msg = f"Incorrect record schema supplied to {self}."
                raise ValueError(msg)
            self.__insert__(record, **kwargs)

    @abc.abstractmethod
    def __append__(self, frame, **kwargs):
        pass

    def append(self, frame, **kwargs):
        """Appends a data frame to table."""
        if not isinstance(frame, dtl.DataLogsBase):
            msg = f"Table append requires a dataframe, not a " + \
                  f"{type(frame)} instance."
            raise TypeError(msg)
        if not isinstance(frame.schema, type(self.schema)):
            msg = f"Incorrect data frame schema supplied to {self}."
            raise ValueError(msg)
        self.__append__(frame, **kwargs)

    def __update__(self, record, **kwargs):
        raise NotImplementedError

    def update(self, *records, **kwargs):
        """Updates rows in place from records, or inserts if not found."""
        try:
            for record in records:
                if not isinstance(record, rcd.Record):
                    msg = f"Table update requires a record, not a " + \
                          f"{type(record)} instance."
                    raise TypeError(msg)
                if not isinstance(record.schema, type(self.schema)):
                    msg = f"Incorrect record schema supplied to {self}."
                    raise ValueError(msg)
                self.__update__(record, **kwargs)
        except NotImplementedError:
            msg = "Updating is not implemented on {0} table objects."""
            raise DataTableWriteException(msg.format(type(self)))

    def __delete__(self, idx, **kwargs):
        raise NotImplementedError

    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        for idx in idxs:
            try:
                self.__delete__(idx, **kwargs)
            except NotImplementedError:
                msg = "Deleting is not implemented on {0} table objects."""
                raise DataTableWriteException(msg.format(type(self)))

    def __record__(self, idx, **kwargs):
        """Returns a raw record data map from its index.

        If the corresponding key is not found, raises an EmptyQueryException.
        """
        raise NotImplementedError

    def record(self, *idx, retry=True, **kwargs):
        """Returns a a record from its index."""
        if len(idx) == 1:
            idx = idx[0]
        try:
            data = self.__record__(idx, **kwargs)
        except NotImplementedError:
            msg = "Fetching records by key is not implemented on {0} objects."
            raise NotImplementedError(msg.format(type(self)))
        except EmptyQueryException:
            msg = f"No row found with index {idx}."
            raise KeyError(msg)
        # Single retry, which seems to eliminate most problems.
        except _Rendezvous:
            if retry:
                return self.record(idx, retry=False, **kwargs)
            raise
        data['index'] = idx
        data['schema'] = self.schema
        return rcd.Record.from_dict(data)

# =============================================================================
# Query base class
# =============================================================================


class DataQueryException(DataTableReadException):
    """Specialized exception type for queries."""
    pass


class EmptyQueryException(DataQueryException):
    """Signals an error due to an empty query."""
    pass


class DataQuery:
    """Abstract base class for data queries.

    Concrete classes implement abstract methods for a particular store type.
    Params:
        table: a DataTable instance.
        *columns: an iterable of column names that will be produced in the
            query results -i.e. the selected columns. Admissible values are
            the column names in the table's schema. If not specified, all
            column are returned. Index column are always returned and should
            not be specified.
        **quargs: query arguments. Admissible keywords are the column names
            from the table's schema, and admissible values must be
            compatible with the range function associated with the
            correponding column, i.e. time_range, float_range or cat_range
            from the nxrange module.
        sql: Additionally, the sql keyword can receive a string which will
            be interpreted as a sql statement. If a value is passed to sql:
            - if other keyword arguments are passed, a DataQueryException is
                raised;
            - if the underlying database engine (e.g. BigTable) doesn't
                support SQL, a DataQueryException is raised.
    """
    # Indicates whether a query type supports sql.
    __sql__ = False

    def __init__(self, table, *columns, exclude=None, sql=None, **quargs):
        self.table = table
        self.schema = type(table.schema)(*columns, exclude=exclude)
        if sql is not None:
            if not self.__sql__:
                msg = f"{self.table} does not support SQL statements."
                raise DataQueryException(msg)
            if quargs:
                msg = "Cannot specify both query arguments and SQL statement."
                raise DataQueryException(msg)
        self.sql = sql

        query_args = []
        for k, v in quargs.items():
            try:
                column = self.table.schema[k]
                assert column.data_range
            except KeyError:
                msg = f"Query argument {k} doesn't match a schema column name."
                raise DataQueryException(msg)
            except AssertionError:
                msg = f"Column {k} doesn't admit selection statements."
                raise DataQueryException(msg)
            query_args.append((column.registration_id,
                               (k, column.data_range(v))))
        query_args.sort()
        self.quargs = OrderedDict([elm[1] for elm in query_args])

    @property
    def id_range(self):
        return self.quargs.get('id', None)

    @property
    def dt_range(self):
        return self.quargs.get('datetime', nxrange.time_range((None, None)))

    @abc.abstractmethod
    def __fetch__(self, **kwargs):
        """Returns an iterator of row indexes and data."""
        return iter([])

    def fetch(self, retry=True, **kwargs):
        """Returns an iterator of query results, as raw data."""
        try:
            for index, data in self.__fetch__(**kwargs):
                yield index, data
        # Single retry, which seems to eliminate most problems.
        except _Rendezvous:
            if retry:
                for index, data in self.fetch(retry=False, **kwargs):
                    yield index, data
            else:
                raise

    def first(self, **kwargs):
        try:
            index, data = next(self.fetch(**kwargs))
        except StopIteration:
            msg = "Query returns no results."
            raise EmptyQueryException(msg)
        data['index'] = index
        data['schema'] = self.schema
        return rcd.Record.from_dict(data)

    def data(self, **kwargs):
        """Returns the data."""
        cls = dtl.archetype(self.id_range, self.dt_range)
        try:
            index, data = zip(*self.fetch(**kwargs))
        except ValueError:
            data = pd.DataFrame(columns=self.schema)
        else:
            id, datetime = zip(*index)
            payload = [elm['data'] for elm in data]
            data = pd.DataFrame(payload)
            data['id'] = id
            data['datetime'] = datetime
        return cls(data, schema=self.schema,
                   id_range=self.id_range, dt_range=self.dt_range)
