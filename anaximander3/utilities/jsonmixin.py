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


class JsonMixin:
    """Mixin class to provide comparison methods.

    Usage requires that classes that implement this mixin define a method
    to_dict and a class method from_dict.
    """

    def json_dumps(self, **kwargs):
        """Serializes self to a json string."""
        kwargs.setdefault('default', serialize)
        return json.dumps(self.to_dict(), **kwargs)

    def json_dump(self, fp, **kwargs):
        """Serializes self to a json file."""
        kwargs.setdefault('default', serialize)
        return json.dump(self.to_dict(), fp, **kwargs)

    @classmethod
    def json_loads(cls, string, **kwargs):
        """Creates an instance from a serialized string."""
        return cls.from_dict(json.loads(string, **kwargs))

    @classmethod
    def json_load(cls, fp, **kwargs):
        """Creates an instance from a file path."""
        return cls.from_dict(json.load(fp, **kwargs))


@singledispatch
def serialize(val):
    return str(val)


@serialize.register(datetime)
def serialize_datetime(t):
    return t.isoformat()


@serialize.register(timedelta)
def serialize_timedelta(d):
    """Serializes to nanoseconds."""
    return int(d.total_seconds() * 1e9)
