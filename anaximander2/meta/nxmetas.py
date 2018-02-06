#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the root mechanisms of Anaximander's type system.

The most critical and complex aspect of this module is the mechanics
necessary to implement the archetype model.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from collections import OrderedDict
import threading

from ..utilities import functions as fun
from . import NxMetaError
from .nxdescriptors import MetaCharacter, TypeProperty

__all__ = []

# Global lock on programmatic subtype creation
LOCK = threading.Lock()

# =============================================================================
# Metaprogramming infrastructure
# =============================================================================


def nxmeta(basetype, basename=None):
    """A metaclass constructor from a supplied archetypical basetype.

    nxmeta creates a metaclass by subclassing the basetype's metaclass
    and modifiying its behavior based on options that may be found
    in an internal Meta class.
    """
    basename = fun.get(basename, basetype.__name__)
    name = basename + 'Type'
    bases = (type(basetype),)
    meta = type(name, bases, {})
    meta.__basename__ = basename
    return meta


class ArchMeta(type):
    """Metaclass for ArcheType mix-in metaclasses."""

    def __init__(mcl, name, bases, namespace):
        # Holds an archetype, the only instance of any ArcheType subclass.
        mcl.__archetype__ = None
        # Subtype registry
        mcl.registry = {}
        if not bases:  # mcl is ArcheType, no initialization needed:
            return
        try:
            basetype = namespace['__basetype__']
            metatype = namespace['__metatype__']
            assert issubclass(metatype, type(basetype))
        except (KeyError, AssertionError):
            msg = "Improper ArchMeta declaration."
            raise NxMetaError(msg)
        # Create registry for metacharacters
        metatype.metacharacters = OrderedDict()


class ArcheType(metaclass=ArchMeta):

    def __new__(mcl, name, bases, namespace, **kwargs):
        """Emulates NxType.__new__ as the arguments are redirected."""
        if mcl is ArcheType:
            msg = "Cannot instantiate abstract metaclass ArcheType."
            raise NxMetaError(msg)
        if mcl.__archetype__ is None:
            # In this case all arguments are ignored and the __archetype__
            # instance is created.
            mcl._swap_metacharacters()
            name = mcl.__basetype__.__name__
            bases = (mcl.__basetype__,)
            archetype = super().__new__(mcl, name, bases, {})
            archetype.__module__ = mcl.__basetype__.__module__
            archetype.__metacharacters__ = mcl.__metatype__.metacharacters
            # Set the archetype on mcl and __metatype__
            mcl.__metatype__.__archetype__ = archetype
            mcl.__archetype__ = archetype
            return archetype
        else:
            cls = mcl.__metatype__(name, bases, namespace, **kwargs)
            return cls

    @classmethod
    def _swap_metacharacters(mcl):
        """Extract metacharacters from base type, swap to typeproperties."""
        basetype, metatype = mcl.__basetype__, mcl.__metatype__
        metacharacters = basetype.nxdescriptors(MetaCharacter)
        for name, char in metacharacters.items():
            typeproperty = TypeProperty(name, char.cls)
            setattr(basetype, name, typeproperty)
            basetype.__nxdescriptors__[name] = typeproperty
            setattr(metatype, name, char)
            metatype.metacharacters[name] = char

    def __init__(archetype, name, bases, namespace):
        archetype.__archetype__ = archetype

    def nxregister(archetype, cls, **kwargs):
        """Registers a subtype."""
        with LOCK:
            cls.__archetype__ = archetype
            type(archetype).registry.register(cls)
            if cls.__bases__[0] is archetype.__basetype__:
                archetype.register(cls)  # registration per abc module.


def archmeta(basetype, basename=None):
    """An Archetype factory function."""
    basename = fun.get(basename, basetype.__name__)
    metatype = nxmeta(basetype, basename)
    name = basename + 'ArcheType'
    bases = (ArcheType, metatype)
    namespace = {'__basetype__': basetype,
                 '__metatype__': metatype}
    return ArchMeta(name, bases, namespace)
