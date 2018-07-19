#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a base metaclass for data objects.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc
from collections import OrderedDict

from ..utilities.jsonmixin import JsonMixin

__all__ = ['DataObjectType', 'DataObject']

# =============================================================================
# Base type
# =============================================================================


class DataObjectType(abc.ABCMeta):
    _registry = dict()  # Type registry

    def __init__(cls, name, bases, namespace):
        if '__schema__' in namespace:
            cls._registry[cls.__schema__] = cls

    def __getitem__(cls, schema):
        if isinstance(schema, type):
            schema_type = schema
        else:
            schema_type = type(schema)
        for stype in schema_type.__mro__:
            try:
                type_ = cls._registry[stype]
                assert issubclass(type_, cls)
                return type_
            except KeyError:
                continue
            except AssertionError:
                raise KeyError
        raise KeyError


class ConformityError(Exception):
    """Raised if data supplied to a DataObject doesn't conform."""
    pass


class DataObject(JsonMixin, metaclass=DataObjectType):
    """Base class for data objects."""
    __schema__ = None  # placeholder for specialized type parameter

    def __new__(cls, data=None, *, schema=None, cast=True, flex=True,
                validate=False, **metadata):
        if schema is not None:
            try:
                type_ = cls[schema]
            except KeyError:
                type_ = cls
        else:
            type_ = cls
        return super().__new__(type_)

    def __init__(self, data=None, *, schema=None, cast=True, flex=True,
                 validate=False, **metadata):
        if schema is not None:
            if isinstance(schema, type):
                self.schema = schema()
            else:
                self.schema = schema
        else:
            self.schema = self.__schema__()
        try:
            assert isinstance(self.schema, self.__schema__)
        except AssertionError:
            msg = f"Improper schema supplied to {type(self).__name__}. " + \
                  f"It must be of type {self.__schema__.__name__}"
            raise ConformityError(msg)
        self._metadata = metadata
        if cast:
            self._data = self.cast(data, flex=flex)
        else:
            self._data = data
        if validate:
            if not self.validate():
                msg = "Validation failed with the following errors: " + \
                      "{self.errors}"
                raise ConformityError(msg)

    @property
    def data(self):
        return self._data.copy()

    @property
    def metadata(self):
        return self._metadata.copy()

    @property
    def columns(self):
        return OrderedDict(self.schema)

    def __getattr__(self, name):
        if name in self.columns:
            return self.data[name]
        else:
            msg = f"'{type(self).__name__}' object has no attribute '{name}'"
            raise AttributeError(msg)

    def __call__(self, *columns, exclude=None):
        """Returns a subset of self with regards to columns."""
        selfcols = list(self.schema.schema_columns)
        if not columns:
            columns_ = selfcols
        else:
            columns_ = []
            for c in columns:
                if c in selfcols:
                    columns_.append(c)
                elif c in self.payload:
                    col = self.schema.payload[c]
                    columns._append('#'.join((col.generic.name, c)))
                else:
                    raise ValueError(f"Unknown columns in {columns}.")
        schema = type(self.schema)(*columns_, exclude=exclude)
        return type(self)(self.data, schema=schema, **self.metadata)
