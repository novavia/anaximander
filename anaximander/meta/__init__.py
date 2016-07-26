#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to Anaximander's meta package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Import statements
#==============================================================================

from abc import ABC, ABCMeta

#==============================================================================
### Abstract base class declarations
#==============================================================================


class nxobject(ABC):
    """Base class for all Anaximander object-like entities.
    
    This includes all types and first-class objects. One notable
    exception is nxdata objects, which are not nxobjects but are
    rather considered interfaces.
    """
    
    def __init__(self, *args, **kwargs):
        self._object_folios = set()

    @property
    def _folios(self):
        """This property is used to hold references to registration folios."""
        return self._object_folios


class meta(ABCMeta):
    """A custom type for Anaximander metaclasses.

    Gotcha: the instances of this class are... metaclasses! This is pushing
    the abstraction of the model to a very unusual extreme and may make it
    difficult to understand right away. However it is worth noting, and
    probably helpful conceptually, that the methods are written using
    the name 'mcl' to refer to instances. For class methods, the designation
    'meta' is used to refer to meta itself.
    Note: this may need to implement a type registry in each newly
    created metatype, i.e. a cladogram.
    """

    def __new__(meta, name, bases, namespace, basename):
        return type.__new__(meta, name, bases, namespace)

    def __init__(mcl, name, bases, namespace, basename):
        """Initializes a new metaclass.
        
        :param basename: base name for types the new metaclass will create.
        """
        mcl.__basename__ = basename

