#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a Mixin class for JSON encoding / decoding.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

from datetime import datetime, timedelta
from functools import singledispatch
import json

import numpy as np

from .functions import monkeypatch


class JsonMixin:
    """Mixin class to provide comparison methods.

    Usage requires that classes that implement this mixin define a method
    to_dict and a class method from_dict.
    """

    def json_dumps(self, **kwargs):
        """Serializes self to a json string."""
        kwargs.setdefault('default', serialize)
        return json.dumps(self.to_dict(**kwargs), **kwargs)

    def json_dump(self, fp, **kwargs):
        """Serializes self to a json file."""
        kwargs.setdefault('default', serialize)
        return json.dump(self.to_dict(**kwargs), fp, **kwargs)

    @classmethod
    def json_loads(cls, string, **kwargs):
        """Creates an instance from a serialized string."""
        return cls.from_dict(json.loads(string, **kwargs), **kwargs)

    @classmethod
    def json_load(cls, fp, **kwargs):
        """Creates an instance from a file path."""
        return cls.from_dict(json.load(fp, **kwargs), **kwargs)


@singledispatch
def serialize(val):
    """Function to supply to the 'default' argument of json dump / dumps."""
    return str(val)


@serialize.register(datetime)
def serialize_datetime(t):
    return t.isoformat()


@serialize.register(timedelta)
def serialize_timedelta(d):
    """Serializes to nanoseconds."""
    return int(d.total_seconds() * 1e9)


@serialize.register(np.int_)
def serialize_npint(i):
    return int(i)


@serialize.register(np.bool_)
def serialize_npbool(b):
    return bool(b)


def jsonio(cls):
    """Decorator to make a type json serializable."""
    monkeypatch(cls, JsonMixin)

    @serialize.register(cls)
    def serialize_(val):
        return val.to_dict()

    return cls
