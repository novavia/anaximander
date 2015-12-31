#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines Anaximander base type and object for libraries and applications.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from .. import nxobject
from .nxmeta import nxmeta

#==============================================================================
### Abstract base classes
#==============================================================================


class NxType(nxmeta, nxobject, metaclass=nxmeta):
    """Parent class to all Anaximander library and application types.

    Note that slots will not work on a derivative of nxobject because of
    the _folios property, but it can work on NxType instances in general.
    """

    def __new__(mcl, name, bases, namespace, patch=None, slots=None, **keys):
        cls = nxmeta.__new__(mcl, name, bases, namespace, patch, slots, **keys)
        cls._cls_folios = set()
        return cls

    @property
    def _folios(cls):
        return cls._cls_folios


class NxObject(nxobject, metaclass=NxType):
    """Anaximander base object class for libraries and applications."""
    pass


class NxDataType(NxType):
    """Base metaclass for Anaximander data types."""
    pass


class nxdata(metaclass=NxDataType):
    """Anaximander base data type."""
    pass
