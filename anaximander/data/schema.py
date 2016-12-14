#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema module for specifying data schemas.

This module patches marshmallow schemas to add column semantics, which
are stored in Fields metadata. These semantics include 'key' and 'serial'.
Similar to database terminology, key fields serve to identify a record
uniquely and are used as indexing values. Serial fields define a natural
order, such that a set of records that otherwise share the same keys can
be organized into series indexed by a serial field -think of a timestamp as
the most typical situation.

Implementation-wise, the module uses monkeypatching on marshmallow schemas.
Hence the Anaximander Schema class is Marshmallow's Schema class, with
a few modifications applied dynamically at runtime. Notable modifications
are as follows:

* Fields have a name property, which is set even in the context of the
Schema class that declares them (this contrasts with Marshmallow's
implementation where the name attribute only exists for fields that are
attribute of a Schema instance).
* Fields have key, serial and description properties, the values for which
are stored in the metadata attribute of the Field.
* Schema implements a fields property, which always return an OrderedDict
whose order follows the declaration sequence of the fields. There is also
a keys property which returns the subset of fields that are tagged as keys.
* Some rules are applied to Fields at Schema instantation:
    * Certain field types are forbidden because they cannot be made
    compatible with the rest of the framework, at least for the time being.
    * Use of marshmallow's missing is disabled and will raise an error. This
    is because records are intended for deserialization and the record
    class makes use of fields' default settings to fill in missing values.
    It is simply more consistent and efficient to only use default, even
    if that may mean some loss of functionality -which has yet to be
    determined.
    * A field that is a key and does not define a default value automatically
    becomes 'required' even if the flag isn't set in the field declaration.
    * All required fields (including keys that don't define a default value)
    must be declared above non-required fields. This is necessary to
    create a record class' init method that lists arguments with no
    default values first.
    * Breaking any of the above rules will raise a SchemaError.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from collections import OrderedDict

import attr
import marshmallow as msh
from marshmallow.fields import Field, Raw, Nested, String, UUID, Number, \
    Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Str, Bool, Int
from marshmallow.schema import SchemaMeta

from ..utilities.functions import monkeypatch
from ..utilities.xprops import cachedproperty, weakproperty


# Compatibility check map
_field_check = {'Field': True,
                'Raw': True,
                'Nested': True,
                'Dict': False,
                'List': False,
                'String': True,
                'UUID': True,
                'Number': True,
                'Integer': True,
                'Decimal': True,
                'Boolean': True,
                'FormattedString': True,
                'Float': True,
                'DateTime': True,
                'LocalDateTime': True,
                'Time': True,
                'Date': True,
                'TimeDelta': True,
                'Url': True,
                'URL': True,
                'Email': True,
                'Method': False,
                'Function': False,
                'Str': True,
                'Bool': True,
                'Int': True,
                'Constant': False,
                }

# =============================================================================
# Marshmallow patching
# =============================================================================


class SchemaError(Exception):
    """A customized error class for schema-related errors."""
    pass

# Pass-through for Marshmallow's ValidationError.
ValidationError = msh.exceptions.ValidationError


class SchemaOpts(msh.SchemaOpts):
    """Sets default options for anaximander schemas.

    These options include:
    * strict is set to True, irrespective or settings in Meta
    * add an option 'record_class_name'
    """
    def __init__(self, meta):
        super().__init__(meta)
        self.strict = True
        self.record_class_name = getattr(meta, 'record_class_name', None)


class Schema(msh.Schema):
    OPTIONS_CLASS = SchemaOpts


class _FieldPatch:
    """Patch of attributes and methods for Field."""
    _name = None  # replaces name in the original implementation.

    @property
    def name(self):
        if self.parent is None:
            return self.metadata.get('name', None)
        else:
            return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def key(self):
        return self.metadata.get('key', False)

    @property
    def serial(self):
        return self.metadata.get('serial', False)

    @property
    def description(self):
        return self.metadata.get('description', None)

    def _attribute_default(self):
        """Extracts the default attribute instantiation value from a field."""
        default = self.default
        if default is msh.missing:
            return None
        elif callable(default):
            return attr.Factory(default)
        else:
            return default

monkeypatch(Field, _FieldPatch)


class _SchemaMetaPatch:
    """Patch of attributes and methods for SchemaMeta."""

    def __init__(cls, name, bases, attrs):
        # Run the regular __init__ method
        cls._schema_meta_init(cls, name, bases, attrs)
        # Mechanism to enforce that required fields are listed first
        required_flag = True
        for k, v in cls.fields.items():
            if k in cls._reserved_names:
                msg = "'{0}' is a reserved name that cannot be used for " + \
                    "schema field names."
                raise SchemaError(msg.format(k))
            v.metadata['name'] = k
            if not _field_check.get(type(v).__name__):
                msg = "Field type {0} for {1} is not supported."
                raise SchemaError(msg.format(type(v).__name__, k))
            if v.missing is not msh.missing:
                msg = "Missing values for deserialization " + \
                    "are not supported by Anaximander Schemas. You may " + \
                    "use a default value, which the corresponding record " + \
                    "will use in lieu of missing. The offending field is " + \
                    "{0} in {1}"
                raise SchemaError(msg.format(k, cls.__name__))
            if v.key is True and v.default is msh.missing:
                v.required = True
            if required_flag is True:
                if not v.required:
                    required_flag = False
            else:
                if v.required:
                    msg = "A required field ({0}) follows non-required " + \
                        "fields."
                    raise SchemaError(msg.format(k))
            setattr(cls, k, v)

    @property
    def base_schemas(cls):
        """Filters bases for Schema subclasses."""
        return tuple(c for c in cls.__bases__ if issubclass(c, Schema))

    @property
    def fields(cls):
        """Returns an OrderedDict of fields, sequenced by creation index."""
        items = sorted(cls._declared_fields.items(),
                       key=lambda i: i[1]._creation_index)
        return OrderedDict(items)

    @property
    def own_fields(cls):
        """Returns non-inherited fields."""
        inherited = msh.schema._get_fields_by_mro(cls, Field, True)
        return OrderedDict(f for f in cls.fields.items() if f not in inherited)

    @property
    def keys(cls):
        """Returns an OrderedDict of key fields in a Schema instance."""
        return OrderedDict((k, v) for k, v in cls.fields.items() if v.key)

    @weakproperty
    def tract(cls):
        """A pointer to a host Tract object, if any."""
        return None

    def set_record_class(cls, record_class):
        """Sets a record class that is called in post-loading."""
        cls.make_record = msh.post_load(lambda i, d: record_class(**d))
        # Re-run processors initialization on the class
        cls._resolve_processors()


# If condition added to enable graceful module reload.
if not hasattr(SchemaMeta, '__patches__'):
    SchemaMeta._schema_meta_init = staticmethod(SchemaMeta.__init__)

monkeypatch(SchemaMeta, _SchemaMetaPatch)
SchemaMeta._reserved_names = set(dir(SchemaMeta) + dir(Schema()))
