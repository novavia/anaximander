#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander base classes for use in applications.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from .meta import NxType, NxObject

#==============================================================================
### Abstract base classes
#==============================================================================


class Type(NxType):
    """The parent metaclass to all Objects."""

    def __new__(mcl, name, bases, namespace, slots=None, **keys):
        if not bases:
            bases = (Object,)
        cls = NxType.__new__(mcl, name, bases, namespace, slots, **keys)
        return cls


class Object(NxObject, metaclass=Type):
    """Parent class to all library and application objects."""
    pass
