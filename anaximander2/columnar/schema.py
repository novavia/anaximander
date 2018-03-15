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


class FieldMapType(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        NxField.collect(cls, namespace)

    @property
    def index_fields(cls):
        return OrderedDict(fun.vfilter(lambda f: f.index, cls.__nxfields__))

    @property
    def sequencer_fields(cls):
        return OrderedDict(fun.vfilter(lambda f: f.sequencer,
                                       cls.__nxfields__))

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

    def __init__(self):
        self._fields = OrderedDict([(k, v.copy())
                                    for k, v in self.__nxfields__.items()])
        for k, v in self._fields.items():
            setattr(self, k, v)
        if not self._fields:
            msg = "Cannot instantiate field-less schema."
            raise SchemaError(msg)

    def __getitem__(self, key):
        return self._fields[key]

    def __iter__(self):
        return self._fields.__iter__()

    def __len__(self):
        return self._fields.__len__()


class SchemaIndexType(FieldMapType):
    """Metaclass for SchemaIndex."""

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        for field in cls.__nxfields__.values():
            field.index = True
        if len(cls.sequencer_fields) > 1:
            msg = "SchemaIndex classes feature at most one sequencer field."
            raise SchemaError(msg)


class SchemaIndex(FieldMap, metaclass=SchemaIndexType):
    """A field map destined to serve as a schema index."""

    def __init__(self):
        super().__init__()
        sequencer_fields = type(self).sequencer_fields
        if sequencer_fields:
            sequencer = list(sequencer_fields.values())[0]
            self._sequencer = self[sequencer.name]
            self._keys = [f for f in self.values() if f is not self._sequencer]
        else:
            self._sequencer = None
            self._keys = list(self.values())

    @property
    def sequencer(self):
        return self._sequencer

    @property
    def keys(self):
        return self._keys

    @abc.abstractmethod
    def rowkey(self, **record):
        """Method to compute the row key from index columns."""
        return None

    @abc.abstractmethod
    def rowloc(self, key):
        """Method to compute index columns from key."""
        return None


class ColumnFamily(FieldMap, NxField):
    """A field map that also serves as a field in a larger schema.

    This is designed to map to column families in columnar databases.
    """

    def __init__(self, name=None, cls=None, registration_id=None):
        NxField.__init__(self, False, name, cls, registration_id)
        super().__init__()


class SchemaBaseType(FieldMapType):

    def __new__(mcl, name, bases, namespace, index=None):
        return super().__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, index=None):
        super().__init__(name, bases, namespace)

        if index is not None:
            if cls.index_fields:
                msg = "A schema class can borrow an index or define index " + \
                      "fields, but not both."
                raise SchemaError(msg)
            cls.__index__ = index
            nxfields = index.__nxfields__.copy()
            nxfields.update(cls.__nxfields__)
            cls.__nxfields__ = nxfields
        else:
            ix_name = name + 'Index'
            ix_namespace = cls.index_fields
            if hasattr(cls, 'rowkey'):
                ix_namespace['rowkey'] = cls.rowkey
            if hasattr(cls, 'rowloc'):
                ix_namespace['rowloc'] = cls.rowloc

            def exec_body(ns):
                ns.update(ix_namespace)
                return ns

            cls.__index__ = types.new_class(ix_name, (SchemaIndex,),
                                            exec_body=exec_body)


class SchemaBase(FieldMap, metaclass=SchemaBaseType):
    """Base class for schemas."""

    def __init__(self):
        self.index = self.__index__()
        self._fields = OrderedDict(self.index._fields)

    @property
    def sequencer(self):
        return self.index._sequencer

    @property
    def keys(self):
        return self.index._keys


class SchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, columns=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, columns=None):
        super().__init__(name, bases, namespace, index=index)

        if columns is not None:
            if cls.column_fields:
                msg = "A schema class can borrow columns or define column " + \
                      "fields, but not both."
                raise SchemaError(msg)
            cls.__columns__ = columns
        else:
            cl_name = name + 'Columns'
            cl_namespace = cls.column_fields

            def exec_body(ns):
                ns.update(cl_namespace)
                return ns

            cls.__columns__ = types.new_class(cl_name, (FieldMap,),
                                              exec_body=exec_body)
        if cls.__columns__.column_fields != cls.__columns__.__nxfields__:
            msg = "Ill-defined schema columns."
            raise SchemaError(msg)


class Schema(SchemaBase, metaclass=SchemaType):

    def __init__(self):
        super().__init__()
        self.columns = self.__columns__()
        self._fields.update(self.columns._fields)


class MultiSchemaType(SchemaBaseType):

    def __new__(mcl, name, bases, namespace, index=None, families=None):
        return super().__new__(mcl, name, bases, namespace, index=index)

    def __init__(cls, name, bases, namespace, index=None, families=None):
        super().__init__(name, bases, namespace, index=index)

        if families is not None:
            if cls.family_fields:
                msg = "A schema class can borrow families or define " + \
                      "family fields, but not both."
                raise SchemaError(msg)
            cls.__families__ = families
            nxfields = index.__nxfields__.copy()
            nxfields.update(cls.__nxfields__)
            cls.__nxfields__ = nxfields
        else:
            fm_name = name + 'Families'
            fm_namespace = cls.family_fields

            def exec_body(ns):
                ns.update(fm_namespace)
                return ns

            cls.__families__ = types.new_class(fm_name, (FieldMap,),
                                               exec_body=exec_body)
        if cls.__families__.family_fields != cls.__families__.__nxfields__:
            msg = "Ill-defined multi-schema family columns."
            raise SchemaError(msg)


class MultiSchema(SchemaBase, metaclass=MultiSchemaType):

    def __init__(self):
        super().__init__()
        self.families = self.__families__()
        self._fields.update(self.families)
