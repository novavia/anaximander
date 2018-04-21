#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a Mixin class for JSON encoding / decoding.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

import json


class JsonMixin:
    """Mixin class to provide comparison methods.

    Usage requires that classes that implement this mixin define a method
    to_dict and a class method from_dict.
    """

    def json_dumps(self, **kwargs):
        """Serializes self to a json string."""
        return json.dumps(self.to_dict(), **kwargs)

    def json_dump(self, fp, **kwargs):
        """Serializes self to a json file."""
        return json.dump(self.to_dict(), fp, **kwargs)

    @classmethod
    def json_loads(cls, string, **kwargs):
        """Creates an instance from a serialized string."""
        return cls.from_dict(json.loads(string, **kwargs))

    @classmethod
    def json_load(cls, fp, **kwargs):
        """Creates an instance from a file path."""
        return cls.from_dict(json.load(fp, **kwargs))
