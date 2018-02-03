#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the root mechanisms of Anaximander's type system.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from collections import OrderedDict

from ..utilities.functions import typecheck
from . import NxMetaError
from .nxdescriptors import MetaCharacter, TypeProperty

__all__ = ['NxMeta', 'nxmeta', 'ArcheType']

# =============================================================================
# NxMeta metaclass type
# =============================================================================


class NxMeta(type):
    """The metaclass for metatypes.

    In this version, it serves no other function than to provide a base
    class for ArchMeta, and thus clarify the metaprogramming levels
    from NxMeta to NxType to NxObject.
    """
    pass


def nxmeta(basetype):
    """A metaclass constructor from a supplied archetypical basetype.

    nxmeta creates a metaclass by subclassing the basetype's metaclass
    and modifiying its behavior based on options that may be found
    in an internal Meta class.
    """
    basename = basetype.__name__
    name = basename + 'Type'
    bases = (type(basetype),)
    return NxMeta(name, bases, {})


class ArchMeta(NxMeta):

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
            assert isinstance(type(basetype), NxMeta)
            assert isinstance(metatype, NxMeta)
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


def archmeta(basetype):
    """An ArcheMeta factory function."""
    metatype = nxmeta(basetype)
    name = basetype.__name__ + 'ArcheType'
    bases = (ArcheType, metatype)
    namespace = {'__basetype__': basetype,
                 '__metatype__': metatype}
    return ArchMeta(name, bases, namespace)
