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
import re

import attr
import marshmallow as msh
from marshmallow.fields import Field, Raw, Nested, Dict, List, String, UUID, \
    Number, Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Method, Function, \
    Str, Bool, Int, Constant
import pandas as pd

from ..utilities.functions import monkeypatch
from .exceptions import DataError, ValidationError as DataValidationError
from .data import NxScalar

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
    def sequential(self):
        return self.metadata.get('sequential', False)

    @property
    def description(self):
        return self.metadata.get('description', None)

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


class NxDataField(Field):
    """Base class for Scalar and Vector fields."""
    pass


class Scalar(NxDataField):
    """A field that expects NxScalar values, whose type is specified."""

    def __init__(self, datatype=NxScalar, default=msh.missing,
                 attribute=None, load_from=None, dump_to=None, error=None,
                 validate=None, required=False, allow_none=None,
                 load_only=False, dump_only=False, missing=msh.missing,
                 error_messages=None, **metadata):
        if not issubclass(datatype, NxScalar):
            raise FieldError("Non NxDataType passed to DataField.")
        self.datatype = datatype
        super().__init__(default=default, attribute=attribute,
                         load_from=load_from, dump_to=dump_to, error=error,
                         validate=validate, required=required,
                         allow_none=allow_none, load_only=load_only,
                         dump_only=dump_only, missing=missing,
                         error_messages=error_messages, **metadata)

    def _serialize(self, value, attr, obj):
        """Expects value to be of type datatype, returns native python type."""
        if not isinstance(value, self.datatype):
            self.fail('type')
        return value.data.item()

    def _deserialize(self, value, attr, data_):
        """Marshals value to a datatype."""
        try:
            return self.datatype(value)
        except DataValidationError:
            self.fail('validator_failed')


class Timestamp(Field):
    """A field that deserializes to a pandas Timestamp."""

    def _serialize(self, value, attr, obj):
        if not isinstance(value, pd.Timestamp):
            self.fail('type')
        return str(value)

    def _deserialize(self, value, attr, data_):
        try:
            return pd.Timestamp(value)
        except ValueError:
            self.fail('validator_failed')


class Duration(Field):
    """A field that deserializes to a pandas Timedelta."""

    def _serialize(self, value, attr, obj):
        if not isinstance(value, pd.Timedelta):
            self.fail('type')
        return str(value)

    def _deserialize(self, value, attr, data_):
        try:
            return pd.Timedelta(value)
        except ValueError:
            self.fail('validator_failed')


class Period(Field):
    """A field that deserializes to a pandas Period of a set frequency."""

    def __init__(self, freq=None, default=msh.missing, attribute=None,
                 load_from=None, dump_to=None, error=None, validate=None,
                 required=False, allow_none=None, load_only=False,
                 dump_only=False, missing=msh.missing, error_messages=None,
                 **metadata):
        try:
            pd.Period(freq=freq)
        except ValueError:
            raise FieldError("Invalid freq specification passed \
                             to Period field.")
        self._freq = freq
        super().__init__(default=default, attribute=attribute,
                         load_from=load_from, dump_to=dump_to, error=error,
                         validate=validate, required=required,
                         allow_none=allow_none, load_only=load_only,
                         dump_only=dump_only, missing=missing,
                         error_messages=error_messages, **metadata)

    @property
    def freq(self):
        return self._freq

    def _serialize(self, value, attr, obj):
        if not isinstance(value, pd.Period):
            self.fail('type')
        return str(value)

    def _deserialize(self, value, attr, data_):
        try:
            return pd.Period(value, freq=self._freq)
        except ValueError:
            self.fail('validator_failed')
