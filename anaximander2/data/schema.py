#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the schema base types.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import Mapping, OrderedDict
import types

from ..utilities import functions as fun, xprops
from .columns import NxColumn


__all__ = []

# =============================================================================
# Base types
# =============================================================================


class SchemaError(Exception):
    """A customized exception for Schema construction errors."""
    pass


class ColumnMapType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        NxColumn.collect(cls, namespace)

    @property
    def index_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index, cls.__nxcolumns__))

    @property
    def sequencer_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index == 'sequential',
                                       cls.__nxcolumns__))

    @property
    def identifier_columns(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index == 'nominal',
                                       cls.__nxcolumns__))

    @property
    def payload_columns(cls):
        columns = OrderedDict()
        for k, v in cls.__nxcolumns__.items():
            if isinstance(v, ColumnFamily):
                continue
            if not v.index:
                columns[k] = v
        return columns

    @property
    def column_families(cls):
        return OrderedDict(fun.vfilter(fun.typecheck(ColumnFamily),
                                       cls.__nxcolumns__))


class ColumnMap(Mapping, metaclass=ColumnMapType):

    def __init__(self):
        self._columns = self.__nxcolumns__.copy()
        for k, v in self._columns.items():
            setattr(self, k, v)
        if not self._columns:
            msg = "Cannot instantiate column-less schema."
            raise SchemaError(msg)

    def __getitem__(self, key):
        return self._columns[key]

    def __iter__(self):
        return self._columns.__iter__()

    def __len__(self):
        return self._columns.__len__()


class SchemaIndexType(ColumnMapType):
    """Metaclass for SchemaIndex."""

    def __new__(mcl, name, bases, namespace):
        if bases[0] == ColumnMap:
            return super().__new__(mcl, name, bases, namespace)
        cmap = ColumnMapType(name, (), namespace)
        identifier = bool(cmap.identifier_columns)
        sequencer = bool(cmap.sequencer_columns)
        if identifier and sequencer:
            return DualSchemaIndexType(name, bases, namespace)
        elif identifier:
            return NominalSchemaIndexType(name, bases, namespace)
        elif sequencer:
            return SequentialSchemaIndexType(name, bases, namespace)
        else:
            return EmptySchemaIndexType(name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if len(cls.identifier_columns) > 1:
            msg = "SchemaIndex classes feature at most one identifier column."
            raise SchemaError(msg)
        if len(cls.sequencer_columns) > 1:
            msg = "SchemaIndex classes feature at most one sequencer column."
            raise SchemaError(msg)
        cls.rowkey = cls.__rowkey__
        cls.rowidx = cls.__rowidx__


class EmptySchemaIndexType(SchemaIndexType):
    """A schema index with no columns."""

    def __new__(mcl, name, bases, namespace):
        return ColumnMapType.__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert not cls.identifier_columns
            assert not cls.sequencer_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifer = None
            cls._sequencer = None


class NominalSchemaIndexType(SchemaIndexType):
    """A schema index with a single, nominal column."""

    def __new__(mcl, name, bases, namespace):
        return ColumnMapType.__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert cls.identifier_columns
            assert not cls.sequencer_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = list(cls.identifier_columns)[0]
            cls._sequencer = None


class SequentialSchemaIndexType(SchemaIndexType):
    """A schema index with a single, sequential column."""

    def __new__(mcl, name, bases, namespace):
        return ColumnMapType.__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert not cls.identifier_columns
            assert cls.sequencer_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = None
            cls._sequencer = list(cls.sequencer_columns)[0]


class DualSchemaIndexType(SchemaIndexType):
    """A schema index with a nominal and a sequential column."""

    def __new__(mcl, name, bases, namespace):
        return ColumnMapType.__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        try:
            assert cls.identifier_columns
            assert cls.sequencer_columns
        except AssertionError:
            msg = "Incorrect index columns specification."
            raise SchemaError(msg)
        else:
            cls._identifier = list(cls.identifier_columns)[0]
            cls._sequencer = list(cls.sequencer_columns)[0]


class SchemaIndex(ColumnMap, metaclass=SchemaIndexType):
    """Base class for SchemaIndex objects."""
    _identifier = None
    _sequencer = None

    @property
    def sequencer(self):
        return self._sequencer

    @property
    def identifier(self):
        return self._identifier

    @abc.abstractmethod
    def __rowkey__(self, index):
        """Method to compute the row key from index tuple."""
        return ""

    @abc.abstractmethod
    def __rowidx__(self, key):
        """Method to compute index columns from key."""
        return ()


class ColumnFamily(ColumnMap, NxColumn):
    """A column map that also serves as a data column in a larger schema.

    This is designed to map to column families in columnar databases.
    """

    def __init__(self, name=None, cls=None, registration_id=None):
        NxColumn.__init__(self, None, name, cls, registration_id)
        super().__init__()


class SchemaBaseType(ColumnMapType):

    def __new__(mcl, name, bases, namespace, index=None):
        return super().__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, index=None):
        super().__init__(name, bases, namespace)

        if index is not None:
            if cls.index_columns:
                msg = "A schema class can borrow an index or define index " + \
                      "columns, but not both."
                raise SchemaError(msg)
            cls.__index__ = index
            nxcolumns = index.__nxcolumns__.copy()
            nxcolumns.update(cls.__nxcolumns__)
            cls.__nxcolumns__ = nxcolumns
        else:
            ix_name = name + 'Index'
            ix_namespace = cls.index_columns
            if hasattr(cls, '__rowkey__'):
                ix_namespace['__rowkey__'] = cls.rowkey
            if hasattr(cls, '__rowidx__'):
                ix_namespace['__rowidx__'] = cls.rowidx

            def exec_body(ns):
                ns.update(ix_namespace)
                return ns

            cls.__index__ = types.new_class(ix_name, (SchemaIndex,),
                                            exec_body=exec_body)


class SchemaBase(ColumnMap, metaclass=SchemaBaseType):
    """Base class for schemas."""

    def __init__(self):
        self.index = self.__index__()
        self._columns = OrderedDict(self.index._columns)

    @property
    def sequencer(self):
        return self.index._sequencer

    @property
    def identifier(self):
        return self.index._identifier

    def rowkey(self, *idx):
        """Method to compute the row key from index tuple."""
        return self.index.rowkey(*idx)

    def rowidx(self, key):
        """Method to compute index columns from key."""
        return self.index.rowidx(key)


class SchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, payload=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, payload=None):
        super().__init__(name, bases, namespace, index=index)

        if payload is not None:
            if cls.payload_columns:
                msg = "A schema class can borrow payload columns or " + \
                      "define payload columns, but not both."
                raise SchemaError(msg)
            cls.__payload__ = payload
        else:
            cl_name = name + 'Payload'
            cl_namespace = cls.payload_columns

            def exec_body(ns):
                ns.update(cl_namespace)
                return ns

            cls.__payload__ = types.new_class(cl_name, (ColumnMap,),
                                              exec_body=exec_body)
        if cls.__payload__.payload_columns != cls.__payload__.__nxcolumns__:
            msg = "Ill-defined schema payload."
            raise SchemaError(msg)


class Schema(SchemaBase, metaclass=SchemaType):

    def __init__(self):
        super().__init__()
        self.payload = self.__payload__()
        self._columns.update(self.payload._columns)


class IndexedColumnType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, column=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, column=None):
        super().__init__(name, bases, namespace, index=index)

        if column is not None:
            if cls.payload_columns:
                msg = "An IndexedColumn type can either declare a column " + \
                      "or be supplied with it but not both."
                raise SchemaError(msg)
            cls.__column__ = column
        else:
            try:
                assert len(cls.payload_columns) == 1
            except AssertionError:
                if name == 'IndexedColumn':
                    return
                msg = "An IndexedColumn type features a single payload " + \
                      "column."
                raise SchemaError(msg)
            cls.__column__ = list(cls.payload_columns.values())[0]


class IndexedColumn(SchemaBase, metaclass=IndexedColumnType):
    """A mini-schema comprising an index and a single column, for series."""

    def __init__(self):
        super().__init__()
        self.column = self.__column__
        self._columns[self.column.name] = self.column


class MultiSchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, families=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, families=None):
        super().__init__(name, bases, namespace, index=index)

        if families is not None:
            if cls.column_families:
                msg = "A schema class can borrow families or define " + \
                      "column families, but not both."
                raise SchemaError(msg)
            cls.__families__ = families
            columns = index.__nxcolumns__.copy()
            columns.update(cls.__nxcolumns__)
            cls.__nxcolumns__ = columns
        else:
            fm_name = name + 'Families'
            fm_namespace = cls.column_families

            def exec_body(ns):
                ns.update(fm_namespace)
                return ns

            cls.__families__ = types.new_class(fm_name, (ColumnMap,),
                                               exec_body=exec_body)
        if cls.__families__.column_families != cls.__families__.__nxcolumns__:
            msg = "Ill-defined multi-schema column families."
            raise SchemaError(msg)


class MultiSchema(SchemaBase, metaclass=MultiSchemaType):

    def __init__(self):
        super().__init__()
        self.families = self.__families__()
        self._columns.update(self.families)
