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

from functools import wraps

import pandas as pd

from ..utilities import nxattr
from ..meta import prototype, metacharacter, newtypemethod, typeinitmethod
from .base import DataObject
from .fields import NxField
from .schema import Schema, SampleLogSchema, EventLogSchema, \
    PhaseLogSchema, ClipLogSchema, FixChartSchema, PostChartSchema, \
    SectionChartSchema, SpanChartSchema

__all__ = ['NxRecord', 'Sample', 'Event', 'Transition', 'Clip',
           'Fix', 'Post', 'Junction', 'Span']

# =============================================================================
# Record base class
# =============================================================================


@prototype
class NxRecord(DataObject):
    schema = metacharacter(validate=lambda s: issubclass(s, Schema))

    @property
    def data(self):
        """Returns a pandas-based view of self, i.e. a Series."""
        return pd.Series(self.as_pytuple(), index=self.schema.fieldnames)

    @classmethod
    def load(cls, data):
        """Loads a record from a serialized data map."""
        return cls.schema().load(data).data

    def dump(self):
        """Serializes a record."""
        return self.schema().dump(self).data

    @classmethod
    def loads(cls, data):
        """Loads a record from a json-serialized data map."""
        return cls.schema().loads(data).data

    def dumps(self):
        """Serializes a record to json."""
        return self.schema().dumps(self).data

    @property
    def keys(self):
        """Return a tuple corresponding to the schema's key fields."""
        return (getattr(self, k) for k in self.schema.keynames)

    def as_dict(self, **kwargs):
        """Returns self's field attributes in dictionary form.

        See attr.asdict for documentation on keyword arguments.
        """
        return nxattr.asdict(self, **kwargs)

    def as_tuple(self, **kwargs):
        """Returns self's field attributes in tuple form.

        See attr.astuple for documentation on keyword arguments.
        """
        return nxattr.astuple(self, **kwargs)

    def as_pydict(self, **kwargs):
        """Returns fields in dict form, converted to native Python."""
        return self.schema.pythonize(self.as_dict(**kwargs))

    def as_pytuple(self, **kwargs):
        """Returns fields in tuple form, converted to native Python."""
        return self.schema.pythonize(self.as_dict(**kwargs), mapping=False)

    @classmethod
    def from_pydict(cls, dict_):
        """Instantiates an object from a dict containing pure Python types.

        Note that this method does not recursively create NxRecords for
        nested fields, but type conversion is applied recursively.
        """
        return cls(**cls.schema.depythonize(dict_))

    @classmethod
    def from_pytuple(cls, tuple_):
        """Instantiates an object from a tuple containing pure Python types.

        Note that this method does not recursively create NxRecords for
        nested fields, but type conversion is applied recursively.
        """
        return cls(**cls.schema.depythonize(tuple_))

    def validate(self):
        """Validates an instance against the schema.

        Note that this is relatively inefficient and not intended for
        production use cases, because the instance has to be serialized,
        and then basically deserialized for validation.
        """
        schema = self.schema()
        schema.validate(schema.dump(self).data)

    @staticmethod
    def _make_attributes(schema):
        """Creates attributes from schema."""
        attrs = {}

        def converter(field):
            """Returns a conversion function."""
            if isinstance(field, NxField):
                return lambda x: field._depythonize(x)
            else:
                return None

        for k, v in schema.fields.items():
            if v.required:
                attrs[k] = nxattr.ib(convert=converter(v))
            else:
                attrs[k] = nxattr.ib(default=v._attribute_default,
                                     convert=converter(v))
        return attrs

    @staticmethod
    def _decorate_init(init):
        """An __init__ method decorator for subtypes.

        The decorator adds two keyword arguments:
        * context, to populate DataObject's weak property context.
        * validate, to optionally validate the inputs. Note that this
            is not done very efficiently in the current implementation.
        """
        @wraps(init)
        def wrapped(*args, **kwargs):
            context = kwargs.pop('context', None)
            validate = kwargs.pop('validate', False)
            init(*args, **kwargs)
            self = kwargs.get('self', args[0])
            self.context = context
            if validate:
                self.validate()
        return wrapped

    @newtypemethod
    def rebase(cls):
        """Rebases the type if its Schema inherits from a base Schema."""
        # cls.schema is Schema or directly inherits from it
        if len(cls.schema.base_schemas) <= 1:
            return cls
        base_schema = cls.schema.base_schemas[0]
        base = NxRecord[base_schema]
        if issubclass(cls, base):
            return cls
        # If cls doesn't inherit from base, we create a new class that does
        new = base.subtype(name=cls.__name__, schema=cls.schema, overtype=True)
        # If additional descriptors exist in cls, they are ported to new
        xattrs = set(cls.__dict__) - set(new.__dict__)
        for k in xattrs:
            setattr(new, k, getattr(cls, k))
        return new

    @newtypemethod
    def set_attributes(cls):
        """Sets the attributes of cls and decorates it with nxattr.s."""
        attributes = cls._make_attributes(cls.schema)
        for k, v in attributes.items():
            setattr(cls, k, v)
        cls = nxattr.s(cls, inherit=False)
        cls.__init__ = cls._decorate_init(cls.__init__)
        return cls

    @typeinitmethod
    def coerce_schema(cls):
        """Coerces the schema to deserialize to cls."""
        cls.schema.set_record_class(cls)

# =============================================================================
# Specialized records
# =============================================================================


class Sample(NxRecord[SampleLogSchema], overtype=True):
    pass


class Event(NxRecord[EventLogSchema], overtype=True):
    pass


class Transition(NxRecord[PhaseLogSchema], overtype=True):
    pass


class Clip(NxRecord[ClipLogSchema], overtype=True):
    pass


class Fix(NxRecord[FixChartSchema], overtype=True):
    pass


class Post(NxRecord[PostChartSchema], overtype=True):
    pass


class Junction(NxRecord[SectionChartSchema], overtype=True):
    pass


class Span(NxRecord[SpanChartSchema], overtype=True):
    pass
