#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the Tract class.

Tract objects hold references to multiple classes that implement
different structures around a common data schema. These include
single record containers as well as containers for tabular data based
on pandas' DataFrame. The primary role of Tract objects is to provide
a namespace with consistent attributes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from functools import partial
import re
import sys

import attr

from . import schema as sch, record as rec

# =============================================================================
# Tract metaclass
# =============================================================================


class TractError(Exception):
    """Base exception type for Tract-related errors."""
    pass


class TractDefinitionError(TractError):
    """Raised when the definition of a Tract fails."""
    pass


class Tract:
    """A container for a Schema and related DataObject classes.

    By convention, a tract's name should be camelcase, as if it was a class.
    This is justified by the fact that the tract primarily serves as
    a namespace to access classes (e.g. Production.Record, Production.Log).
    """

    def __init__(self, schema, name=None):
        """Initializes a Tract.

        attrs:
            schema: a Schema *class*
            name (str): an optional name for self, which is inserted into
                its module's global namespace. If no name is supplied, it
                is assembled automatically from the Schema class name,
                provided it follows the naming convention [CamelCase]Schema.
        """
        # First check that there is a proper schema
        if not issubclass(schema, sch.Schema):
            msg = "A tract must feature a Schema."
            raise TractDefinitionError(msg)
        self.Schema = schema
        schema.tract = self

        # Assembles name if needed
        if name is None:
            try:
                name = re.match('\w+(?=Schema)', schema.__name__).group(0)
            except AttributeError:
                msg = "The schema has a non-standard name. Either change " + \
                    "its name to end with 'Schema' or supply a custom " + \
                    "name for the target Data class."
                raise TractDefinitionError(msg)
        self._name = name
        # Insert self in the schema's global namespace with name
        sys.modules[schema.__module__].__dict__[name] = self

        # Makes and sets the record class
        self.set_record_class(self._make_record_class())

    @property
    def name(self):
        return self._name

    @property
    def bases(self):
        """Returns 'base' tracts from Schema's bases."""
        base_schemas = self.Schema.base_schemas
        return tuple(s.tract for s in base_schemas if s.tract is not None)

    def _bases(self, type_):
        """Returns the base classes for a given DataObject type.

        attrs:
            type_: a DataObject base type such as Log or Record.

        returns:
            a tuple of bases for the passed object type.
        """
        name = type_.__name__
        bases = (getattr(b, name, None) for b in self.bases)
        return tuple(b for b in bases if issubclass(b, type_)) or (type_,)

    def _make_record_attributes(self):
        """Extract a list of attribute specifications from Schema class."""
        schema = self.Schema
        attrs = {}
        for k, v in schema.fields.items():
            if v.required:
                attrs[k] = attr.ib()
            else:
                attrs[k] = attr.ib(default=v._attribute_default())
        # Special attribute _validate makes it possible to add a 'validate'
        # option to the __init__ method of the record class, while ignoring
        # it for most practical purposes.
        validate = attr.ib(False, repr=False, cmp=False, hash=False)
        attrs['_validate'] = validate
        return attrs

    def _make_record_class(self, name=None):
        """Creates a record class to assign to cls."""
        name = name or self.name + 'Record'
        bases = self._bases(rec.Record)
        attributes = self._make_record_attributes()
        kls = rec.RecordType.auto_init(name, attributes)
        record_class = attr.s(kls)
        record_class.__bases__ = bases
        return record_class

    def set_record_class(self, record_class):
        """Sets the record class."""
        if not issubclass(record_class, rec.Record):
            raise TractDefinitionError()
        self.Record = record_class
        record_class.tract = self
        self.Schema.set_record_class(record_class)


def tract(cls=None, *, name=None):
    """The tract decorator, which picks up a Tract from a Schema class.

    This decorator offers the most straightforward way to define a Tract.
    Params:
        cls: A Schema declaration.
        name: An optional name for the Tract. If not supplied,
            the function expects that the schema class has a name
            in the form [CamelCase]Schema and will extract CamelCase as the
            name of the Tract.

    Raises:
        TractDefinitionError: if the supplied cls is not a subclass of Schema,
            or if its name doesn't follow conventions and no name is supplied.

    Returns:
        The decorated class.
    """
    # Enables the decorator to function with or without arguments.
    if cls is None:
        return partial(tract, name=name)
    Tract(cls, name=name)
    return cls
