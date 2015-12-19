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

from ..arche import apeiron, nxobject

#==============================================================================
### Abstract base classes
#==============================================================================


class NxMeta(apeiron, nxobject):
    """The base metaclass to all Anaximander Types and Objects."""

    @classmethod
    def __baptize__(mcl, bases, **kwargs):
        """A class namer, from bases and kwargs. Provided as placeholder."""
        raise TypeError

    @classmethod
    def __prepare__(mcl, name, bases, **kwargs):
        return {}

    def __new__(mcl, name, bases, namespace, **kwargs):
        cls = apeiron.__new__(mcl, name, bases, namespace)
        cls._cls_folios = set()
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        apeiron.__init__(cls, name, bases, namespace)

    @property
    def _folios(cls):
        return cls._cls_folios


class NxBaseType(NxMeta, metaclass=NxMeta):
    """Parent class to all Anaximander types."""
    pass


class NxBaseObject(nxobject, metaclass=NxBaseType):
    """Base class for all Anaximander objects."""
    pass
