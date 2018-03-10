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

from ..utilities import functions as fun
from ..fields import NxField


__all__ = []

# =============================================================================
# Base types
# =============================================================================


class SchemaError(Exception):
    """A customized exception for Schema construction errors."""
    pass


class FieldMapType(abc.ABCMeta, Mapping):

    def __init__(cls, name, bases, namespace):
        NxField.collect(cls, namespace)

    def __getitem__(cls, key):
        return cls.__nxfields__[key]

    def __iter__(cls):
        return cls.__nxfields__.__iter__()

    def __len__(cls):
        return cls.__nxfields__.__len__()

    @property
    def index_fields(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index, cls.__nxfields__))

    @property
    def column_fields(cls):
        columns = OrderedDict()
        for k, v in cls.__nxfields__.items():
            if isinstance(v, ColumnFamily):
                continue
            if not v.index:
                columns[k] = v
        return columns

    @property
    def family_fields(cls):
        return OrderedDict(fun.vfilter(fun.typecheck(ColumnFamily),
                                       cls.__nxfields__))


class FieldMap(Mapping, metaclass=FieldMapType):

    def __getitem__(self, key):
        return self.__nxfields__[key]

    def __iter__(self):
        return self.__nxfields__.__iter__()

    def __len__(self):
        return self.__nxfields__.__len__()


class Index(FieldMap):
    """A field map destined to serve as a schema index."""

    @abc.abstractproperty
    def rowkey(self):
        """Method to compute the row key from index columns."""
        return None


class ColumnFamily(FieldMap, NxField):
    """A field map that also serves as a field in a larger schema.

    This is designed to map to column families in columnar databases.
    """
    pass

    @property
    def index(self):
        return False


class SchemaBaseType(FieldMapType):

    def __new__(mcl, name, bases, namespace, index=None):
        return super().__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, index=None):
        super().__init__(name, bases, namespace)

        if index is not None:
            if any(f.index for f in cls.values()):
                msg = "A schema class can borrow an index or define index " + \
                      "fields, but not both."
                raise SchemaError(msg)
            cls.__index__ = index
        else:
            ix_name = name + 'Index'
            ix_namespace = cls.index_fields
            if hasattr(cls, 'rowkey'):
                ix_namespace['rowkey'] = cls.rowkey

            def exec_body(ns):
                ns.update(ix_namespace)
                return ns

            cls.__index__ = types.new_class(ix_name, (Index,),
                                            exec_body=exec_body)


class SchemaBase(FieldMap, metaclass=SchemaBaseType):
    """Base class for schemas."""
    pass


class SchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, columns=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, columns=None):
        super().__init__(name, bases, namespace, index=index)


class Schema(SchemaBase, metaclass=SchemaType):
    pass


class MultiSchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, families=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, families=None):
        super().__init__(name, bases, namespace, index=index)


class MultiSchema(SchemaBase, metaclass=MultiSchemaType):
    pass
