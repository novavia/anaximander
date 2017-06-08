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
import warnings

import pandas as pd

from ..utilities import xprops, nxrange
from ..meta import NxObject, typeinitmethod, archetype, typeattribute
from .frame import NxDataFrame, NxDataCollection, NxDataSequence
from .annotations import interval

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


class DataTableWarning(UserWarning):
    """Customized warning type for data tables."""
    pass

# =============================================================================
# Class declaration
# =============================================================================


class DataTable(NxObject):
    """Abstract base class for data storage resources.

    Basically a light wrapper around table objects defined by various
    database APIs to unify basic commands.
    """
    schema = None  # placeholder, a metacharacter in derived prototypes

    def __hash__(self):
        return id(self)

    @abc.abstractmethod
    def __create__(self, **kwargs):
        pass

    @abc.abstractproperty
    def exists(self):
        pass

    def create(self, warn=True, remove=None, **kwargs):
        """Creates the table in the database.

        Params:
            warn (Bool): if True, a warning and confirmation prompt are raised
                if the table already exists.
            remove: if None, proceed to prompt the user in case an existing
                table needs to be removed. If T/F, skips the prompt and
                operate accordingly.
        """
        if warn and self.exists:
            msg = "Attempting to create table {0}, which already exists."
            warnings.warn(msg.format(self), DataTableWarning)
            if remove is None:
                if self.remove() is False:
                    return
            elif remove is True:
                self.remove(False)
            elif remove is False:
                return
        return self.__create__(**kwargs)

    @abc.abstractmethod
    def __remove__(self):
        pass

    def remove(self, confirm=True, **kwargs):
        """Removes the table in the database.

        Params:
            confirm: a confirmation prompt designed to prevent accidental
                mistakes.
        """
        if confirm:
            msg = "This operation will delete {0} and all its data. " + \
                  "Type 'YES' if you wish to proceed."
            confirmation = input(msg.format(self))
            if not confirmation == 'YES':
                print("Table remove operation aborted.")
                return False
        return self.__remove__(**kwargs)

    @abc.abstractmethod
    def __query__(self, *fields, **kwargs):
        pass

    def query(self, *fields, **kwargs):
        """Returns a Query object."""
        return self.__query__(*fields, **kwargs)

    @abc.abstractmethod
    def __insert__(self, record, **kwargs):
        pass

    def insert(self, *records, **kwargs):
        """Inserts one or more records into table."""
        for record in records:
            self.__insert__(record, **kwargs)

    @abc.abstractmethod
    def __append__(self, frame, **kwargs):
        pass

    def append(self, frame, **kwargs):
        """Appends a data frame to table."""
        if not isinstance(frame, NxDataFrame) or frame.schema != self.schema:
            msg = "Incorrect data frame type supplied to {0}."
            raise TypeError(msg.format(self))
        self.__append__(frame, **kwargs)

    def __update__(self, record, **kwargs):
        raise NotImplementedError

    def update(self, *records, **kwargs):
        """Updates rows in place from records, or inserts if not found."""
        try:
            for record in records:
                self.__update__(record, **kwargs)
        except NotImplementedError:
            msg = "Updating is not implemented on {0} table objects."""
            raise DataTableWriteException(msg.format(type(self)))

    def __delete__(self, rowkey, **kwargs):
        raise NotImplementedError

    def delete(self, *rowkeys, **kwargs):
        """Deletes the specified rows from a key or other arguments."""
        try:
            for rowkey in rowkeys:
                self.__delete__(rowkeys, **kwargs)
        except NotImplementedError:
            msg = "Deleting is not implemented on {0} table objects."""
            raise DataTableWriteException(msg.format(type(self)))

# =============================================================================
# Query base class
# =============================================================================


class DataQueryException(DataTableReadException):
    """Specialized exception type for queries."""
    pass    


class EmptyQueryException(DataQueryException):
    """Signals an error due to an empty query."""
    pass


@archetype
class DataQuery(NxObject):
    """Abstract base class for data queries.

    Concrete classes implement abstract methods for a particular store type.
    Params:
        table: a DataTable instance.
        *fields: an iterable of field names that will be produced in the
            query results -i.e. the selected columns. Admissible values are
            the field names in the table's schema. If not specified, all
            fields are returned. Key fields are always returned and do not
            need to be specified.
        **quargs: query arguments. Admissible keywords are the field names
            from the table's schema, and admissible values are of three
            types with strict conventions:
            - scalar values are interpreted as an equal statement;
            - tuples are interpreted as a range statement and converted into
                an interval (from data.nxrange);
            - lists are interpreted as an 'is in' statement.
        sql: Additionally, the sql keyword can receive a string which will
            be interpreted as a sql statement. If a value is passed to sql:
            - if other keyword arguments are passed, a DataQueryException is
                raised;
            - if the underlying database engine (e.g. BigTable) doesn't
                support SQL, a DataQueryException is raised.
    """
    # Indicates whether a query type supports sql.
    __sql__ = typeattribute(False)

    def __init__(self, table, *fields, sql=None, **quargs):
        self.table = table
        schemafields = self.table.schema.fields
        def fieldsortkey(name):
            return schemafields[name]._creation_index
        if not all(f in table.schema.fieldnames for f in fields):
            msg = "All requested query fields must match a schema field name."
            raise DataQueryException(msg)
        if not fields:
            self.fields = table.schema.fieldnames
        else:
            fields = set(fields) | set(self.table.schema.keynames)
            self.fields = tuple(sorted(fields, key=fieldsortkey))
        if sql is not None:
            if quargs:
                msg = "Queries cannot specifiy both sql and query arguments."
                raise DataQueryException(msg)
            else:
                try:
                    self.sql = sql
                except AttributeError:
                    msg = "{0} table type does not support SQL statements."
                    raise DataQueryException(msg.format(type(self.table)))                    
        query_args = []

        for k, v in quargs.items():
            try:
                field = schemafields[k]
            except KeyError:
                msg = "All query arguments must match a schema field name."
                raise DataQueryException(msg)                
            if isinstance(v, nxrange.Range):
                query_args.append((k, v))
            elif isinstance(v, tuple):
                query_args.append((k, interval(*v, ref=field)))
            else:
                query_args.append((k, nxrange.levels(v)))
        query_args.sort(key=lambda p: fieldsortkey(p[0]))
        self.quargs = OrderedDict(query_args)

    @xprops.singlesetproperty
    def sql(self):
        return self._sql_generator()

    def _sql_generator(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __fetch__(self, **kwargs):
        """Returns an iterator of rows as tuples."""
        return iter([])

    def fetch(self, **kwargs):
        """Returns an iterator of query results, as rows."""
        return self.__fetch__(**kwargs)

    def first(self, **kwargs):
        try:
            row = dict(zip(self.fields, next(self.fetch(**kwargs))))
        except StopIteration:
            msg = "Query returns no results."
            raise EmptyQueryException(msg)
        schema = self.table.schema()
        return schema.load(row).data

    @property
    def frametype(self):
        return NxDataCollection[self.table.schema].frametype

    @property
    def sequencetype(self):
        return NxDataSequence[self.table.schema]

    def data(self, **kwargs):
        """Returns a pandas DataFrame with the results."""
        return pd.DataFrame(self.__fetch__(**kwargs), columns=self.fields)

    def frame(self, context=None, **kwargs):
        """Returns query results as a data frame."""
        return self.frametype(self.data(**kwargs), context=context)

    def sequence(self, context=None, **kwargs):
        """Returns query results as a data frame.
        
        Note: this requires that the data contains a single non-sequential
        key, which is not enforced by the method.
        """
        return self.sequencetype(self.data(**kwargs), context=context)

    def records(self, **kwargs):
        """Returns results as a list of records."""
        return self.frame(**kwargs).to_records()

    @typeinitmethod
    def _set_sql_property(cls):
        if cls.__sql__:
            cls.sql = xprops.singlesetproperty(lambda s: s._sql_generator())
        else:
            cls.sql = property(lambda s: NotImplemented)
        