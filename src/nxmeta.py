#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander base type and object classes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from abc import ABC, ABCMeta

#==============================================================================
### Abstract base classes
#==============================================================================


class NxMeta(ABC):
    """Abstract base class to all Anaximander Types and Objects."""

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj._nxregistries = set()
        return obj


class NxType(NxMeta, ABCMeta):
    """Parent class to all Anaximander types."""

    def __new__(mcl, name, bases, namespace, **kwargs):
        cls = type.__new__(mcl, name, bases, namespace)
        cls._nxregistries = set()
        return cls


class NxObject(NxMeta, metaclass=NxType):
    """Base class for all Anaximander objects."""
    pass
