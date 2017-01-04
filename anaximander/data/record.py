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

import attr

from ..utilities import nxattr
from ..meta.metadescriptors import MetaCharacter, newtypemethod, typeinitmethod
from ..meta.nxtype import archetype
from ..meta.nxobject import NxObject
from .schema import Schema

# =============================================================================
# Record base class
# =============================================================================


@archetype
class Record(NxObject):

    schema = MetaCharacter(validate=lambda s: issubclass(s, Schema))

    @property
    def data(self):
        return self.as_dict()

    @classmethod
    def load(cls, data):
        """Loads a record from a serialized data map."""
        return cls.schema().load(data).data

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

    @classmethod
    def attr_filter(cls, filter):
        """Composes a filter with the class' base filter."""
        return lambda a, v: False if a.name == '_validate' else filter(a, v)

    def __attrs_post_init__(self):
        """Additional __init__ procedure -see attrs doc. for details."""
        try:
            validate = self._validate
        except AttributeError:
            pass  # Fails silently
        else:
            if validate:
                self.validate()
            del self._validate

    @staticmethod
    def _make_attributes(schema):
        """Creates attributes from schema."""
        attrs = {}
        for k, v in schema.fields.items():
            if v.required:
                attrs[k] = nxattr.ib()
            else:
                attrs[k] = nxattr.ib(default=v._attribute_default())
        # Special attribute _validate makes it possible to add a 'validate'
        # option to the __init__ method of the record class, while ignoring
        # it for most practical purposes.
        validate = nxattr.ib(False, repr=False, cmp=False, hash=False)
        attrs['_validate'] = validate
        return attrs

    @newtypemethod
    def set_attributes(cls):
        """Sets the attributes of cls and decorates it with nxattr.s."""
        attributes = cls._make_attributes(cls.schema)
        for k, v in attributes.items():
            setattr(cls, k, v)
        return nxattr.s(cls, inherit=False)

    @typeinitmethod
    def coerce_schema(cls):
        """Coerces the schema to deserialize to cls."""
        cls.schema.set_record_class(cls)
