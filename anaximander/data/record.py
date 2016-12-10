#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a base Record class.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import abc

import attr

from ..utilities.xprops import weakproperty

# =============================================================================
# Record base class
# =============================================================================


class RecordType(abc.ABCMeta):
    """Metaclass for Record classes."""

    def __init__(cls, name, bases, attrs):
        auto = attrs.pop('__auto__', False)
        if auto:
            cls.tract = None
        # Ensures that a Data tract's Record class is updated if that
        # class is subclassed.
        elif cls.tract is not None:
            cls.tract.set_record_class(cls)

    @classmethod
    def auto_init(mcl, name, bases, attrs):
        """Init method used by Tract objects for automatic creation.

        This enables traceability so we can distinguish automated creation
        from a purpose-built Record class that defines additional methods.
        """
        attrs['__auto__'] = True
        return mcl(name, bases, attrs)

    @weakproperty
    def tract(self):
        """Pointer to the owning Tract object."""
        return None


class Record(abc.ABC, metaclass=RecordType):
    """Abstract base class for record classes."""

    _validate = False  # Placeholder, replaced by concrete implementations.

    @property
    def schema(self):
        """Returns the schema type associated with self."""
        return type(self).tract.Schema

    @classmethod
    def load(cls, data):
        """Loads a record from a serialized data map."""
        return cls.tract.Schema().load(data).data

    def dump(self):
        """Serializes a record."""
        return self.schema().dump(self).data

    def as_dict(self, **kwargs):
        """Returns self's field attributes in dictionary form.

        See attr.asdict for documentation on keyword arguments.
        """
        filter = kwargs.pop('filter', lambda a, v: True)
        kwargs['filter'] = self.attr_filter(filter)
        return attr.asdict(self, **kwargs)

    def as_tuple(self, **kwargs):
        """Returns self's field attributes in tuple form.

        See attr.astuple for documentation on keyword arguments.
        """
        filter = kwargs.pop('filter', lambda a, v: True)
        kwargs['filter'] = self.attr_filter(filter)
        return attr.astuple(self, **kwargs)

    def validate(self):
        """Validates an instance against the schema.

        Note that this is relatively inefficient and not intended for
        production use cases, because the instance has to be serialized,
        and then basically deserialized for validation.
        """
        schema = self.schema()
        schema.validate(schema.dump(self).data)

    def __attrs_post_init__(self):
        """Additional __init__ procedure -see attrs doc. for details."""
        if self._validate:
            self.validate()

    @classmethod
    def attr_filter(cls, filter):
        """Composes a filter with the class' base filter."""
        return lambda a, v: False if a is cls._validate else filter(a, v)
