#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schema module for specifying data schemas.

This module patches marshmallow schemas to add column semantics, which
are stored in Fields metadata. These semantics include 'key' and 'sequential'.
Similar to database terminology, key fields serve to identify a record
uniquely and are used as indexing values. Sequential fields define a natural
order, such that a set of records that otherwise share the same keys can
be organized into series indexed by a sequential field -think of a timestamp
as the most typical situation.

Implementation-wise, the module uses monkeypatching on marshmallow schemas.
Hence the Anaximander Schema class is Marshmallow's Schema class, with
a few modifications applied dynamically at runtime. Notable modifications
are as follows:

* Fields have a name property, which is set even in the context of the
Schema class that declares them (this contrasts with Marshmallow's
implementation where the name attribute only exists for fields that are
attribute of a Schema *instance*).
* Fields have key, sequential and description properties, the values for
which are stored in the metadata attribute of the Field.
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

import marshmallow as msh
from marshmallow.schema import SchemaMeta

from ..utilities.functions import monkeypatch
from ..utilities.xprops import weakproperty, cachedproperty
from .exceptions import DataError
from . import fields
from .data import NxFloat

__all__ = ['Schema', 'SchemaError', 'SampleLogSchema', 'EventLogSchema',
           'PhaseLogSchema', 'ClipLogSchema', 'FixChartSchema',
           'PostChartSchema', 'SectionChartSchema', 'SpanChartSchema']

# =============================================================================
# Schema patching
# =============================================================================


class SchemaError(DataError):
    """A customized error class for schema-related errors."""
    pass

# Pass-through for Marshmallow's ValidationError.
ValidationError = msh.exceptions.ValidationError


class SchemaOpts(msh.SchemaOpts):
    """Sets default options for anaximander schemas.

    These options include:
    * strict is set to True, irrespective or settings in Meta
    * ordered is set to True
    """
    def __init__(self, meta):
        super().__init__(meta)
        self.strict = True
        self.ordered = True


class Schema(msh.Schema):
    OPTIONS_CLASS = SchemaOpts


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
            if not fields.supported(type(v)):
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
        seqkeys = {k for k, v in cls.keys.items() if v.sequential}
        if len(seqkeys) > 1:
            msg = "A schema cannot feature more than one sequential key field."
            raise SchemaError(msg)
        elif len(seqkeys) == 1:
            cls._seqkey = seqkeys.pop()

    @cachedproperty
    def base_schemas(cls):
        """Filters mro for Schema subclasses."""
        return tuple(c for c in cls.__mro__ if issubclass(c, Schema))[1:]

    @cachedproperty
    def fields(cls):
        """Returns an OrderedDict of fields, sequenced by creation index."""
        items = sorted(cls._declared_fields.items(),
                       key=lambda i: i[1]._creation_index)
        return OrderedDict(items)

    @cachedproperty
    def fieldnames(cls):
        """Returns a tuple of field names, in sequence."""
        return tuple(cls.fields.keys())

    @cachedproperty
    def own_fields(cls):
        """Returns non-inherited fields."""
        inherited = msh.schema._get_fields_by_mro(cls, fields.Field, True)
        return OrderedDict(f for f in cls.fields.items() if f not in inherited)

    @cachedproperty
    def keys(cls):
        """Returns an OrderedDict of key fields in a Schema."""
        return OrderedDict((k, v) for k, v in cls.fields.items() if v.key)

    @cachedproperty
    def keynames(cls):
        """Returns a tuple of key field names, in sequence."""
        return tuple(cls.keys().keys())

    @cachedproperty
    def nskeys(cls):
        """Returns an OrderedDict of non-sequential key fields in a Schema."""
        pairs = ((k, v) for k, v in cls.keys.items() if not v.sequential)
        return OrderedDict(pairs)

    @cachedproperty
    def seqkey(cls):
        """Returns the name of the only sequential key or None.

        This is set in the __init__ method as there has to be an integrity
        check.
        """
        return None

    @cachedproperty
    def nonkeyfields(cls):
        """Returns an OrdererdDict of non-key fields."""
        return OrderedDict((k, v) for k, v in cls.fields.items() if not v.key)

    @cachedproperty
    def nonkeyfieldnames(cls):
        """Returns a tuple of non-key field names, in sequence."""
        return tuple(cls.nonkeyfields.keys())

    @cachedproperty
    def required(cls):
        """Returns a dict of required fields in a Schema."""
        return OrderedDict((k, v) for k, v in cls.fields.items() if v.required)

    @weakproperty
    def tract(cls):
        """A pointer to a host DataTract object, if any."""
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


class _SchemaPatch:
    """Adds pythonize / depythonize class methods."""

    @classmethod
    def _keyed_field_extractor(cls, obj, extract, mapping=True):
        collector = []
        for f in cls.fields.values():
            extractor = getattr(f, extract)
            try:
                val = extractor(obj, mapping=mapping)
            except AttributeError:
                continue
            else:
                collector.append(val)
        return collector

    @classmethod
    def _listed_field_extractor(cls, obj, extract, mapping=True):
        collector = []
        for i, f in enumerate(cls.fields.values()):
            extractor = getattr(f, extract)
            try:
                val = extractor(obj, key=i, mapping=mapping)
            except AttributeError:
                continue
            else:
                collector.append(val)
        return collector

    @classmethod
    def pythonize(cls, obj, mapping=True):
        """Returns a mapping or tuple of fields to 'pythonized' values.

        This is applicable where fields deserialize to non-native Python
        types.
        obj is either an object or mapping that implements the cls schema,
        or alternatively a tuple, list where the values are expected to map to
        the schema's fields.
        """
        if isinstance(obj, (tuple, list)):
            collector = cls._listed_field_extractor(obj, 'pygetattr', mapping)
        else:
            collector = cls._keyed_field_extractor(obj, 'pygetattr', mapping)
        if mapping:
            return dict(collector)
        else:
            return tuple(collector)

    @classmethod
    def depythonize(cls, obj, mapping=True):
        """Returns a mapping or tuple of fields to 'depythonized' values.

        This is applicable where fields deserialize to non-native Python
        types.
        """
        if isinstance(obj, (tuple, list)):
            collector = cls._listed_field_extractor(obj, 'ypgetattr', mapping)
        else:
            collector = cls._keyed_field_extractor(obj, 'ypgetattr', mapping)
        if mapping:
            return dict(collector)
        else:
            return tuple(collector)

monkeypatch(Schema, _SchemaPatch)


SchemaMeta._reserved_names = set(dir(SchemaMeta) + dir(Schema()))

# =============================================================================
# Specialized schemas
# =============================================================================


class TimeSchema(Schema):
    """Base class for schemas indexed with timestamps."""
    timestamp = fields.Timestamp(key=True, sequential=True)


class SampleLogSchema(TimeSchema):
    """Base schema class for sample logs.

    Sample logs have a primary function to list sampled values along
    a time axis.
    """
    pass


class EventLogSchema(TimeSchema):
    """Base schema class for event logs.

    Event logs list timestamped events. The events are typed with
    an string code, which may be uniform if there is a single type of
    events.
    """
    event_type = fields.Str(required=True)


class PhaseLogSchema(TimeSchema):
    """Base schema class for phase transition logs.

    Phase logs record changes in state, using string identifier.
    Hence a time range-based query on the log allows to reconstruct phases
    spent in different states across the entire range.
    """
    prev_state = fields.Str(required=True)
    next_state = fields.Str(required=True)


class ClipLogSchema(TimeSchema):
    """Base schema class for clip logs.

    Clip logs require an offset (pandas.DateOffset), which is specified
    in the timestamp field (of type Period). It defaults to hourly.
    """
    timestamp = fields.Period(freq='H', key=True, sequential=True)

    @property
    def freq(self):
        return self.timestamp.freq

    @property
    def freqstr(self):
        return self.timestamp.freqstr


class LinearSchema(Schema):
    """Base class for schemas indexed with a univariate coordinate."""
    coordinate = fields.Scalar(NxFloat, key=True, sequential=True)


class FixChartSchema(LinearSchema):
    """Base class for for fix charts.

    Fix charts simply provide data records at various locations.
    """
    pass


class PostChartSchema(LinearSchema):
    """Base class for post charts.

    Post charts list typed post along a linear axis.
    """
    post_type = fields.Str(required=True)


class SectionChartSchema(LinearSchema):
    """Base class for section charts.

    Section charts record transition between different types of sections.
    """
    prev_type = fields.Str(required=True)
    next_type = fields.Str(required=True)


class SpanChartSchema(LinearSchema):
    """Base class for span charts.

    Span charts require a stride, which is specified in the
    coordinate field (of type TBD, such as MilePost).
    """
    pass
