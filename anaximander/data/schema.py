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
import re
import types

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
# Record base class
# =============================================================================


#TODO: May need to ensure that the _validate attribute is present?
class Record:
    """Abstract base class for record classes."""

    # weak pointer to the schema class
    @weakproperty
    def schema(self):
        return None

    @classmethod
    def load(self, data):
        """Loads a record from a serialized data map."""
        return self.schema().load(data).data

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
#        cls.set_record_class(cls._make_record_class())

    @property
    def base_schemas(cls):
        """Filters bases for Schema subclasses."""
        return tuple(c for c in cls.__bases__ if issubclass(c, Schema))

    @weakproperty
    def Data(cls):
        """A pointer to a host Data class, if any."""
        return None

    @cachedproperty
    def record_class(cls):
        return None

    @property
    def record_class_name(cls):
        return cls._record_class.__name__

    def _make_attributes(cls):
        """Extract a list of attribute specifications from a schema."""
        attrs = {}
        for k, v in cls.fields.items():
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

# TODO: make meaningful doc string for record class
    def _make_record_class(cls):
        """Makes a record class from schema.

        The record class uses attr.make_class to automate the creation
        of boilerplate methods.
        The name of the record class is set automatically as follows:
        * the method checks whether the class defines an option
        record_class_name and uses it if so.
        * Otherwise if the schema class follows the pattern [CamelCase]Schema,
        then the name of the record class is [CamelCase].
        * Otherwise a SchemaError is raised.
        """
        if cls.opts.record_class_name is not None:
            name = cls.opts.record_class_name
        else:
            try:
                name = re.match('\w+(?=Schema)', cls.__name__).group(0)
            except AttributeError:
                if cls.__name__ == 'Schema':
                    return Record
                msg = "The schema has a non-standard name. Either change " + \
                    "the name or supply a record_class_name option in Meta."
                raise SchemaError(msg)

        def body(ns):
            """Populates the class' namespace with field attributes."""
            attributes = cls._make_attributes()
            ns.update(attributes)

        kls = types.new_class(name, (Record,), exec_body=body)
        record_class = attr.s(kls)
        return record_class

    def set_record_class(cls, record_class):
        """Sets the record class associated with the Schema cls."""
        cls._record_class = record_class
        record_class.schema = cls
        cls.make_record = msh.post_load(lambda i, d: record_class(**d))
        # Re-run processors initialization on the class
        cls._resolve_processors()

    @property
    def fields(cls):
        """Returns an OrderedDict of fields, sequenced by creation index."""
        items = sorted(cls._declared_fields.items(),
                       key=lambda i: i[1]._creation_index)
        return OrderedDict(items)

    @cachedproperty
    def keys(cls):
        """Returns an OrderedDict of key fields in a Schema instance."""
        return OrderedDict((k, v) for k, v in cls.fields.items() if v.key)

# If condition added to enable graceful module reload.
if not hasattr(SchemaMeta, '__patches__'):
    SchemaMeta._schema_meta_init = staticmethod(SchemaMeta.__init__)

monkeypatch(SchemaMeta, _SchemaMetaPatch)
SchemaMeta._reserved_names = set(dir(SchemaMeta) + dir(Schema()))
