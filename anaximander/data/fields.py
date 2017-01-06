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
from marshmallow.fields import Field, Raw, Nested, String, UUID, Number, \
    Integer, Decimal, Boolean, FormattedString, Float, DateTime, \
    LocalDateTime, Time, Date, TimeDelta, Url, URL, Email, Str, Bool, Int

from ..utilities.functions import monkeypatch

# =============================================================================
# Utilities
# =============================================================================


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


def supported(field_type):
    """Returns True if field_type is supported in Anaximander.

    Raises TypeError if field_type is not a subclass of Field.
    Returns False if it is but is unsupported. If field_type is not
    found in _field_check, the return value is True as the field_type is
    assumed to be a purpose-built custom subclass.
    """
    if issubclass(field_type, Field):
        return _field_check.get(field_type.__name__, True)
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


# =============================================================================
# Additional Field classes
# =============================================================================


class ReString(String):
    """A String field that gets validated with a regular expression.

    The __init__ method takes a pattern argument that gets compiled into
    the regular expression. Any match validates an input string.
    """

    def __init__(self, default=msh.missing, attribute=None, load_from=None,
                 dump_to=None, error=None, pattern='.*', validate=None,
                 required=False, allow_none=None, load_only=False,
                 dump_only=False, missing=msh.missing, error_messages=None,
                 **metadata):
        self._pattern = pattern
        self._re = re.compile(pattern)
        if validate is None:
            validate = self.match
        elif isinstance(validate, Iterable):
            validate = [self.match] + list(validate)
        else:
            validate = [self.match] + [validate]
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
