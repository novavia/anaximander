#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines abstract base classes for the data package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc

from ..utilities.xprops import weakproperty

# =============================================================================
# Metaclasses
# =============================================================================


class DataObjectType(abc.ABCMeta):
    """Metaclass for DataObject classes."""
    pass


class SchemedDataType(DataObjectType):
    """Metaclass for DataObject classes that implement schemas.

    These include Record, Frame and their derivatives.
    """
    __archetype__ = None

    def __init__(cls, name, bases, attrs):
        auto = attrs.pop('__auto__', False)
        if auto:
            cls.tract = None
        # Ensures that a Data tract's object class is updated if that
        # class is subclassed.
        elif cls.tract is not None:
            cls.tract.set_class(cls)

    @classmethod
    def auto_init(mcl, name, attrs):
        """Init method used by Tract objects for automatic creation.

        This enables traceability so we can distinguish automated creation
        from a purpose-built Record class that defines additional methods.
        """
        attrs['__auto__'] = True
        return mcl(name, (mcl.__archetype__,), attrs)

    @weakproperty
    def tract(cls):
        """Pointer to the owning Tract object."""
        return None

    @property
    def schema(cls):
        """Returns the schema type associated with cls."""
        try:
            return cls.tract.Schema
        except AttributeError:
            try:
                return cls._schema
            except AttributeError:
                return None

    @schema.setter
    def schema(cls, sch):
        """Sets the schema if no tract has been defined.

        This feature mostly serves testing purposes, allowing to
        utilize DataObject classes without defining tracts.
        """
        if cls.tract is not None:
            msg = "Cannot set schema on a DataType associated with a Tract."
            raise AttributeError(msg)
        cls._schema = sch

    @schema.deleter
    def schema(cls):
        if cls.tract is not None:
            msg = "Cannot delete schema on a DataType associated with a Tract."
            raise AttributeError(msg)
        try:
            del cls._schema
        except AttributeError:
            pass

# =============================================================================
# Base classes
# =============================================================================


class DataObject(abc.ABC, metaclass=DataObjectType):
    """Base class for all DataObjects."""

    @abc.abstractproperty
    def data(self):
        """Returns the read-only data contained by the DataObject."""
        pass


class SchemedDataObject(DataObject, metaclass=SchemedDataType):
    """Base class for schema-based DataObjects."""

    @property
    def schema(self):
        """Returns the schema type associated with self."""
        return type(self).schema

    @abc.abstractmethod
    def validate(self):
        """Validates an existing instance against the schema."""
        pass
