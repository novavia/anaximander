#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fields module that provides built-in field types.

This module patches marshmallow fields. See the schema module for explaining
documentation.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from collections import Iterable
from math import isnan
import re

import attr
import marshmallow as msh
from marshmallow.utils import get_value as msh_get_value
from marshmallow.fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant
import pandas as pd

from ..utilities import xprops
from ..utilities.functions import monkeypatch, get
from ..utilities.nxtime import datetime
from .exceptions import DataError, ValidationError as DataValidationError
from .data import NxScalar

MARSHMALLOW_VERSION = int(msh.__version__[0])

if MARSHMALLOW_VERSION == 2:
    get_value = msh_get_value
elif MARSHMALLOW_VERSION == 3:
    def get_value(key, obj, default=msh.missing):
        return msh_get_value(obj, key, default=default)

__all__ = ['Field', 'Nested', 'String', 'UUID', 'Number', 'Integer',
           'Decimal', 'Boolean', 'FormattedString', 'Float', 'DateTime',
           'LocalDateTime', 'Time', 'Date', 'TimeDelta', 'Url', 'URL',
           'Email', 'Str', 'Bool', 'Int', 'ReString', 'Scalar', 'Timestamp',
           'Duration', 'Period', 'FieldError']

# =============================================================================
# Utilities
# =============================================================================

# Compatibility check map
_field_check = {Field: True,
                Raw: True,
                Nested: True,
                Dict: False,
                List: False,
                String: True,
                UUID: True,
                Number: True,
                Integer: True,
                Decimal: True,
                Boolean: True,
                FormattedString: True,
                Float: True,
                DateTime: True,
                LocalDateTime: True,
                Time: True,
                Date: True,
                TimeDelta: True,
                Url: True,
                URL: True,
                Email: True,
                Method: False,
                Function: False,
                Str: True,
                Bool: True,
                Int: True,
                Constant: False,
                }


def supported(field_type):
    """Returns True if field_type is supported in Anaximander.

    Raises TypeError if field_type is not a subclass of Field.
    Returns False if it is but is unsupported. If field_type is not
    found in _field_check, the return value is True as the field_type is
    assumed to be a purpose-built custom subclass.
    """
    if issubclass(field_type, Field):
        return _field_check.get(field_type, True)
    raise TypeError("Non Field subclass supplied to supported.")

# =============================================================================
# Field patching
# =============================================================================


class _FieldPatch:
    """Patch of attributes and methods for Field."""
    _name = None  # replaces name in the original implementation.
    __interval__ = None  # placeholder for specifiying interval function

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
        """Specifies whether a field acts as a key within the host schema.

        Values are typically True/False, however a string can be passed that
        specifies how to design a unique table row key from the key fields.
        Such values can be 'hash', 'reverse' (hashes field values, or
        rewrites them in reverse, respectively), and 'timestamp' for
        UNIX timestamp conversion. See gcbigtable module for details.
        """
        return self.metadata.get('key', False)

    @property
    def sequential(self):
        return self.metadata.get('sequential', False)

    @property
    def description(self):
        return self.metadata.get('description', None)

    @property
    def family(self):
        """An optional column family.

        By convention this is always 'keys' for key fields (any metadata
        specifying otherwise is ignored). For non-key fields, it is looked
        up in metadata and defaults to 'values'.
        """
        if self.key:
            return 'keys'
        return self.metadata.get('family', 'values')

    @property
    def _attribute_default(self):
        """Extracts the default attribute instantiation value from a field."""
        default = self.default
        if default is msh.missing:
            return None
        elif callable(default):
            return attr.Factory(default)
        else:
            return default

    # Placeholder for pandas preprocessing function
    pdpreprocessor = None

    def pygetattr(self, obj, key=None, mapping=True):
        """Extracts (key, val) corresponding to self from obj, 'pythonized'.

        attrs:
            obj: an object presumed to feature an attribute or key
                corresonding to self
            key: optional key, such as an integer if obj is a tuple
            mapping: whether to return results as a mapping or a tuple
                of values.
        """
        attr = self.attribute or self.name
        key = key if key is not None else attr
        try:
            val = get_value(key, obj)
        except TypeError:  # integer key beyond bonds
            val = msh.missing
        if val is msh.missing:
            if self.required:
                msg = "Required field missing from object {0}"
                raise msh.exceptions.ValidationError(msg.format(obj))
            else:
                # Intended to be intercepted at the schema level
                raise AttributeError
        if isinstance(self, Nested):
            pyval = self.schema.pythonize(val, mapping=mapping)
        else:
            pyval = self._pythonize(val)
        if mapping:
            return (attr, pyval)
        else:
            return pyval

    def ypgetattr(self, obj, key=None, mapping=True):
        """Reverse operation of pythonize, see corresponding doc."""
        attr = self.attribute or self.name
        key = key if key is not None else attr
        try:
            val = get_value(key, obj)
        except TypeError:  # integer key beyond bonds
            val = msh.missing
        if val is msh.missing:
            if self.required:
                msg = "Required field missing from object {0}"
                raise msh.exceptions.ValidationError(msg.format(obj))
            else:
                # Intended to be intercepted at the schema level
                raise AttributeError
        if isinstance(self, Nested):
            pyval = self.schema.depythonize(val, mapping=mapping)
        else:
            pyval = self._depythonize(val)
        if mapping:
            return (attr, pyval)
        else:
            return pyval

    def _pythonize(self, val):
        """Turns a deserialized field value to a pure Python value.

        This is an identitiy function by default, and can be overriden
        by subclasses that deserialize to non-native Python.
        """
        return val

    def _depythonize(self, val):
        """Reverse operation of pythonize (see corresponding doc.)"""
        return val

monkeypatch(msh.fields.Field, _FieldPatch)


class FieldError(DataError):
    """Customized exception raised for incorrect Field instantiation."""
    pass

# =============================================================================
# Additional Field classes
# =============================================================================


class ReString(String):
    """A String field that gets validated with a regular expression.

    The __init__ method takes a pattern argument that gets compiled into
    the regular expression. Any match validates an input string.
    """

    def __init__(self, pattern='.*', default=msh.missing, attribute=None,
                 load_from=None, dump_to=None, error=None, validate=None,
                 required=False, allow_none=None, load_only=False,
                 dump_only=False, missing=msh.missing, error_messages=None,
                 **metadata):
        self._pattern = pattern
        try:
            self._re = re.compile(pattern)
        except TypeError:
            raise FieldError("The pattern of a ReString must be a string.")
        if validate is None:
            validate = [lambda s: self.match(s)]
        elif isinstance(validate, Iterable):
            validate = [lambda s: self.match(s)] + list(validate)
        else:
            validate = [lambda s: self.match(s)] + [validate]
        super().__init__(default=default, attribute=attribute,
                         load_from=load_from, dump_to=dump_to, error=error,
                         validate=validate, required=required,
                         allow_none=allow_none, load_only=load_only,
                         dump_only=dump_only, missing=missing,
                         error_messages=error_messages, **metadata)

    @property
    def pattern(self):
        return self._pattern

    def match(self, s):
        """Returns True if string s matches self's pattern, False otherwise."""
        return bool(self._re.match(s))

# Alias
ReStr = ReString


class NxField(Field):
    """Base class for custom Anaximander fields."""
    pass


class NxDataField(NxField):
    """Base class for Scalar and Vector fields."""
    pass


class Scalar(NxDataField):
    """A field that expects NxScalar values, whose type is specified."""
    __defaults__ = {'f': float('nan'),
                    'i': -1,
                    'u': -1,
                    'b': False,
                    'U': '',
                    'S': ''
                    }

    def __init__(self, datatype=NxScalar, default=None,
                 attribute=None, load_from=None, dump_to=None, error=None,
                 validate=None, required=False, allow_none=True,
                 load_only=False, dump_only=False, error_messages=None,
                 **metadata):
        if not issubclass(datatype, NxScalar):
            raise FieldError("Non NxDataType passed to DataField.")
        self.datatype = datatype
        if default is None:
            default = get(default, self.__defaults__[self.kind])
            if metadata.get('key', False):
                required = True
        super().__init__(default=default, attribute=attribute,
                         load_from=load_from, dump_to=dump_to, error=error,
                         validate=validate, required=required,
                         allow_none=allow_none, load_only=load_only,
                         dump_only=dump_only, error_messages=error_messages,
                         **metadata)

    @xprops.cachedproperty
    def kind(self):
        return self.datatype.dtype.kind

    def _serialize(self, value, attr, obj):
        """Expects value to be of type datatype, returns native python type."""
        if value is None:
            return self._attribute_default
        if not isinstance(value, self.datatype):
            self.fail('type')
        return value.data.item()

    def _deserialize(self, value, attr, data_):
        """Marshals value to a datatype."""
        if value is 'None':
            return self._attribute_default
        try:
            return self.datatype(value)
        except DataValidationError:
            self.fail('validator_failed')

    def _pythonize(self, val):
        rval = self.datatype.number(val)
        if isnan(rval):
            return None
        else:
            return rval

    def _depythonize(self, val):
        if val is None:
            return self._attribute_default
        else:
            return self.datatype(val)


class Timestamp(NxField):
    """A field that deserializes to a pandas Timestamp."""

    def __init__(self, default=pd.NaT, attribute=None, load_from=None,
                 dump_to=None, error=None, validate=None, required=False,
                 allow_none=True, load_only=False, dump_only=False,
                 missing=msh.missing, error_messages=None, **metadata):
            super().__init__(default=default, attribute=attribute,
                             load_from=load_from, dump_to=dump_to, error=error,
                             validate=validate, required=required,
                             allow_none=allow_none, load_only=load_only,
                             dump_only=dump_only, missing=missing,
                             error_messages=error_messages, **metadata)

    def _serialize(self, value, attr, obj):
        if value in (None, pd.NaT, 'NaT'):
            return 'NaT'
        if not isinstance(value, pd.Timestamp):
            self.fail('type')
        return str(value)

    def _deserialize(self, value, attr, data_):
        try:
            return datetime(value)
        except ValueError:
            self.fail('validator_failed')

    def _pythonize(self, val):
        if val is not None:
            return val.to_pydatetime(warn=False)

    def _depythonize(self, val):
        if val is not None:
            return datetime(val)


class Duration(NxField):
    """A field that deserializes to a pandas Timedelta."""

    def _serialize(self, value, attr, obj):
        if value is None:
            return msh.missing
        if not isinstance(value, pd.Timedelta):
            self.fail('type')
        return value.value

    def _deserialize(self, value, attr, data_):
        try:
            return pd.Timedelta(value)
        except ValueError:
            self.fail('validator_failed')

    # XXX: this is not technically correct. In practice, the only application
    # of pythonize is conversion of records for BigQuery inserts. Duration
    # requires a conversion to an integer and that is what this function
    # returns, even though a more rigorous definition would have it return
    # a python time delta object.
    def _pythonize(self, val):
        # if val is not None:
        # return val.to_pytimedelta()
        if val is not None:
            return val.value

    def _depythonize(self, val):
        if val is not None:
            return pd.Timedelta(val)


class Period(NxField):
    """A field that deserializes to a pandas Period of a set frequency."""

    def __init__(self, freq=None, default=msh.missing, attribute=None,
                 load_from=None, dump_to=None, error=None, validate=None,
                 required=False, allow_none=None, load_only=False,
                 dump_only=False, missing=msh.missing, error_messages=None,
                 **metadata):
        try:
            period = pd.Period('1970-1-1', freq=freq)
        except ValueError:
            raise FieldError("Invalid freq specification passed \
                             to Period field.")
        self._freq = period.freq
        self._freqstr = period.freqstr
        super().__init__(default=default, attribute=attribute,
                         load_from=load_from, dump_to=dump_to, error=error,
                         validate=validate, required=required,
                         allow_none=allow_none, load_only=load_only,
                         dump_only=dump_only, missing=missing,
                         error_messages=error_messages, **metadata)

    @property
    def freq(self):
        return self._freq

    @property
    def freqstr(self):
        return self._freqstr

    def _serialize(self, value, attr, obj):
        if value is None:
            return msh.missing
        if not isinstance(value, pd.Period):
            self.fail('type')
        return str(value)

    def _deserialize(self, value, attr, data_):
        try:
            return pd.Period(value, freq=self._freq)
        except ValueError:
            self.fail('validator_failed')

    def _pythonize(self, val):
        if val is not None:
            return val.to_timestamp().to_pydatetime(warn=False)

    def _depythonize(self, val):
        if val is not None:
            return pd.Period(val, freq=self._freq)
