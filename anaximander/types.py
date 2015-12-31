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

from ._meta import NxType, NxObject

#==============================================================================
### Abstract base classes
#==============================================================================


class Type(NxType):
    """The parent metaclass to all Objects."""

    def __new__(mcl, name, bases, namespace, patch=None, slots=None, **keys):
        if not bases:
            bases = (Object,)
        cls = NxType.__new__(mcl, name, bases, namespace, patch, slots, **keys)
        return cls


class ClassMethod(object):
    "Emulate PyClassMethod_Type() in Objects/funcobject.c"

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, cls=None):
        f = lambda *a, **kw: self.f(cls, *a, **kw)
        f.__name__ = self.f.__name__
        f.__doc__ = self.f.__doc__
        return f


class Object(NxObject, metaclass=Type):
    """Parent class to all library and application objects."""

    @ClassMethod
    def om(cls):
        print(cls.__name__)

    @classmethod
    def om2(cls):
        print(cls.__name__)
