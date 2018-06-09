#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the Store, Title and Tract classes used for storage.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc
from collections import OrderedDict, Mapping

import attr
import pandas as pd

from ..utilities import nxrange as rge, xprops, functions as fun
from ..utilities.datastore import StorageResource
from ..data.nxschema import Schema, MultiSchema
from ..data import datalogs as dtl, records as rcd


__all__ = ['StorageException', 'StorageAdminException',
           'ReadException', 'WriteException', 'QueryException',
           'EmptyQueryException', 'Store', 'Title', 'Tract']

# =============================================================================
# Custom exceptions
# =============================================================================


class StorageException(Exception):
    """Exception related to data stores."""
    pass


class StorageAdminException(StorageException):
    """Exception related to storage administration."""
    pass


class ReadException(StorageException):
    """Exception related to reading data from a tract."""
    pass


class WriteException(StorageException):
    """Exception related to writing / updating data in a tract."""
    pass

# =============================================================================
# Class declarations
# =============================================================================


@attr.s
class Title:
    """A wrapper for a name / schema association.

    The name is used by stores for naming storage resources.
    Schema can be either an instance or type, but gets converted to an instance
    if the latter.
    """
    name = attr.ib()
    schema = attr.ib(validator=attr.validators.instance_of((Schema,
                                                            MultiSchema)),
                     convert=lambda s: s() if isinstance(s, type) else s)


class Store(Mapping):
    """A light wrapper for data stores with a mapping interface.

    The Store provides a mapping from Title name to Tract.
    """

    def __init__(self, **kwargs):
        self._io = self.__interface__(**kwargs)
        self._tracts = dict()

    @abc.abstractmethod
    def __interface__(self, **kwargs):
        return NotImplemented

    @property
    def io(self):
        """Interface to a storage client library."""
        return self._io

    def __getitem__(self, key):
        return self._tracts.__getitem__(key)

    def __iter__(self):
        return self._tracts.__iter__()

    def __len__(self):
        return self._tracts.__len__()


class Tract(StorageResource):
    """Abstract base class for data storage resources.

    Basically a light wrapper around table objects defined by various
    database APIs to unify basic commands.
    """

    def __init__(self, store, title, register=True):
        self.store = store
        self.title = title
        if register:
            store._tracts[self.name] = self

    @xprops.weakproperty
    def store(self):
        return None

    @property
    def name(self):
        return self.title.name

    @property
    def schema(self):
        return self.title.schema

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return fun.iformat('store', 'name')(self)

    @abc.abstractmethod
    def __query__(self, *columns, **kwargs):
        pass

    def query(self, *columns, **kwargs):
        """Returns a Query object."""
        return self.__query__(*columns, **kwargs)


class DataTract(Tract):
    """A Tract that allows flexible operations on time series data."""

    @abc.abstractmethod
    def __insert__(self, record, **kwargs):
        pass

    def insert(self, *records, **kwargs):
        """Inserts one or more records into tract."""
        inserts = []
        for record in records:
            if not isinstance(record, rcd.Record):
                msg = f"Tract insert requires a record, not a " + \
                      f"{type(record)} instance."
                raise TypeError(msg)
            if not isinstance(record.schema, type(self.schema)):
                msg = f"Incorrect record schema supplied to {self}."
                raise ValueError(msg)
            inserts.append(self.__insert__(record, **kwargs))
        return inserts

    @abc.abstractmethod
    def __append__(self, logs, **kwargs):
        pass

    def append(self, logs, **kwargs):
        """Appends a data logs to tract."""
        if not isinstance(logs, dtl.DataLogsBase):
            msg = f"Tract append requires a datalogs instance, not a " + \
                  f"{type(logs)} instance."
            raise TypeError(msg)
        if not isinstance(logs.schema, type(self.schema)):
            msg = f"Incorrect data schema supplied to {self}."
            raise ValueError(msg)
        return self.__append__(logs=logs, **kwargs)

    def __update__(self, record, **kwargs):
        raise NotImplementedError

    def update(self, *records, **kwargs):
        """Updates rows in place from records, or inserts if not found."""
        updates = []
        try:
            for record in records:
                if not isinstance(record, rcd.Record):
                    msg = f"Tract update requires a record, not a " + \
                          f"{type(record)} instance."
                    raise TypeError(msg)
                if not isinstance(record.schema, type(self.schema)):
                    msg = f"Incorrect record schema supplied to {self}."
                    raise ValueError(msg)
                updates.append(self.__update__(record, **kwargs))
        except NotImplementedError:
            msg = "Updating is not implemented on {0} tract objects."""
            raise WriteException(msg.format(type(self)))
        return updates

    def __delete__(self, idx, **kwargs):
        raise NotImplementedError

    def delete(self, *idxs, **kwargs):
        """Deletes the specified rows based on their index."""
        deletes = []
        for idx in idxs:
            try:
                deletes.append(self.__delete__(idx, **kwargs))
            except NotImplementedError:
                msg = "Deleting is not implemented on {0} tract objects."""
                raise WriteException(msg.format(type(self)))
        return deletes

    def __discard__(self, id, dt_range, **kwargs):
        raise NotImplementedError

    def discard(self, *ids, datetime=(None, None), **kwargs):
        """Deletes ranges of rows within dt_range for specified ids."""
        dt_range = rge.time_range(datetime)
        discards = []
        for id in ids:
            try:
                discards.append(self.__discard__(id, dt_range, **kwargs))
            except NotImplementedError:
                msg = "Discarding is not implemented on {0} tract objects."""
                raise WriteException(msg.format(type(self)))
        return discards

    def __record__(self, idx, **kwargs):
        """Returns a raw record data map from its index.

        If the corresponding key is not found, raises an EmptyQueryException.
        """
        raise NotImplementedError

    def record(self, *idx, **kwargs):
        """Returns a record from its index."""
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
        data['index'] = idx
        data['schema'] = self.schema
        return rcd.Record.from_dict(data)


# =============================================================================
# Query base class
# =============================================================================


class QueryException(ReadException):
    """Specialized exception type for queries."""
    pass


class EmptyQueryException(QueryException):
    """Signals an error due to an empty query."""
    pass


class Query:
    """Abstract base class for data queries.

    Concrete classes implement abstract methods for a particular store type.
    Params:
        tract: a Tract instance.
        *columns: an iterable of column names that will be produced in the
            query results -i.e. the selected columns. Admissible values are
            the column names in the tract's schema. If not specified, all
            column are returned. Index column are always returned and should
            not be specified.
        **quargs: query arguments. Admissible keywords are the column names
            from the tract's schema, and admissible values must be
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

    def __init__(self, tract, *columns, exclude=None, sql=None,
                 metadata=None, **quargs):
        self.tract = tract
        self.schema = type(tract.schema)(*columns, exclude=exclude)
        if sql is not None:
            if not self.__sql__:
                msg = f"{self.tract} does not support SQL statements."
                raise QueryException(msg)
            if quargs:
                msg = "Cannot specify both query arguments and SQL statement."
                raise QueryException(msg)
        self.sql = sql

        query_args = []
        for k, v in quargs.items():
            try:
                column = self.tract.schema[k]
                assert column.data_range
            except KeyError:
                msg = f"Query argument {k} doesn't match a schema column name."
                raise QueryException(msg)
            except AssertionError:
                msg = f"Column {k} doesn't admit selection statements."
                raise QueryException(msg)
            query_args.append((column.registration_id,
                               (k, column.data_range(v))))
        query_args.sort()
        self.quargs = OrderedDict([elm[1] for elm in query_args])
        self.metadata = fun.get(metadata, {})

    @property
    def id_range(self):
        return self.quargs.get('id', None)

    @property
    def dt_range(self):
        return self.quargs.get('datetime', rge.time_range((None, None)))

    @abc.abstractmethod
    def __fetch__(self, **kwargs):
        """Returns an iterator of row indexes and data."""
        return iter([])

    def fetch(self, **kwargs):
        """Returns an iterator of query results, as raw data."""
        for index, data in self.__fetch__(**kwargs):
            yield index, data

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
                   id_range=self.id_range, dt_range=self.dt_range,
                   **self.metadata)
