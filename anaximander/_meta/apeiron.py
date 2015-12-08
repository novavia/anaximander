#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander base metaclass, type and object classes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================


#==============================================================================
### Abstract base classes
#==============================================================================


apeiron = type  # The unbounded origin of all concepts and things.


class nxobject(object):
    """The base class of all Anaximander entitites (meta, type & object)."""
    pass


class NxMeta(apeiron, nxobject):
    """The base metaclass to all Anaximander Types and Objects."""

    @classmethod
    def __prepare__(mcl, name, bases, **kwargs):
        return {}

    def __new__(mcl, name, bases, namespace, **kwargs):
        cls = type.__new__(mcl, name, bases, namespace)
        cls._nxregistries = set()
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        type.__init__(cls, name, bases, namespace)


class NxBaseType(NxMeta, metaclass=NxMeta):
    """Parent class to all Anaximander types."""
    pass


class NxBaseObject(nxobject, metaclass=NxBaseType):
    """Base class for all Anaximander objects."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._nxregistries = set()
        return obj
