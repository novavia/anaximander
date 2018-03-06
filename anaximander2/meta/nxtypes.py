#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines NxType, the root anaximander type.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from collections import ChainMap
from inspect import getmodule
import sys
import types

from ..utilities import functions as fun, xprops
from . import NxMetaError
from . import nxdescriptors as nxd
from .nxmetas import NxMeta, archmeta, ArcheType

__all__ = ['NxType', 'nxtype', 'archetype']

# =============================================================================
# NxMeta metaclass type
# =============================================================================


class NxType(abc.ABCMeta, metaclass=NxMeta):
    __basename__ = 'Nx'  # The basename, usable for metaclass instances.

    @classmethod
    def __baptize__(mcl, basetype, traits=None, **kwargs):
        """Generates a name for a programatically generated type."""
        try:
            basename = kwargs.get('basename', basetype.__basename__)
        except (IndexError, AttributeError):
            msg = "Programmatic naming requires a base name."
            raise NxMetaError(msg)
        return basename

    def __new__(mcl, name, bases, namespace, traits=None, **kwargs):
        # Checks that at most the first base is an NxType
        # Multiple inheritance is supported for mixin classes only
        if any(isinstance(b, NxType) for b in bases[1:]):
            msg = "NxType does not support multiple inheritance."
            raise NxMetaError(msg)
        # If bases[0] is an archetype, we replace it with the basetype
        archetype = mcl.__archetype__
        if archetype is not None:
            base = bases[0]
            # If the base is the archetype, we replace it with __basetype__
            basetype = archetype.__basetype__
            if base is archetype:
                base = basetype
            elif not issubclass(base, basetype):
                msg = "Incorrect use of an ArcheType subclass."
                raise NxMetaError(msg)
            bases = (base,) + bases[1:]
        # Set proto-class
        cls = super().__new__(mcl, name, bases, namespace)
        # Set the type attributes
        # They are either supplied by the namespace, kwargs or inheritance
        attrs = ChainMap(namespace, kwargs)
        for k, v in mcl.typeattributes.items():
            if k in attrs:
                setattr(cls, k, attrs[k])
            else:
                setattr(cls, v.cache, getattr(cls, v.cache, None))
        # Run new type methods in order
        for k in type(cls).nxm_metadescriptors(nxd.NewTypeMethod):
            method = getattr(type(cls), k, None)
            if method:
                cls = method(cls)
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        # Processes metadescriptors and nxdescriptors declared in the namespace
        nxd.MetaDescriptor.collect(cls, namespace)
        nxd.NxDescriptor.collect(cls, namespace)
        archetype = cls.archetype
        overtype = kwargs.get('overtype', False)
        if archetype is not None:
            overtype = kwargs.get('overtype', False)
            archetype.nxregister(cls, overtype=overtype)
        # Run type init methods in order
        for k in type(cls).nxm_metadescriptors(nxd.TypeInitMethod):
            method = getattr(type(cls), k, None)
            if method:
                method(cls)

    def nxdescriptors(cls, *types):
        """Returns a mapping of descriptors filtered by types.

        if no types are supplied, then all registered descriptors are
        returned.
        """
        return nxd.NxDescriptor.retrieve(cls, *types)

    def metadescriptors(cls, *types):
        """Returns a mapping of metadescriptors filtered by types.

        if no types are supplied, then all registered metadescriptors are
        returned.
        """
        return nxd.MetaDescriptor.retrieve(cls, *types)

    @property
    def archetype(cls):
        return type(cls).__archetype__

    @property
    def is_archetype(cls):
        return isinstance(cls, ArcheType)

    @property
    def typeattributes(cls):
        """Tuple of type properties for the archetype's type attributes."""
        return tuple(getattr(cls, k) for k in type(cls).typeattributes)

    @xprops.cachedproperty
    def typeparameters(cls):
        """Tuple of type properties for the archetype's type parameters."""
        return tuple(getattr(cls, k) for k in type(cls).typeparameters)

    @xprops.cachedproperty
    def registration_key(cls):
        """The key with which cls is registered in its metaclass.

        This can return None (no registration), a singular value, or a tuple
        of type parameter values that are declared as keys.
        """
        if cls.is_pending_archetype:
            return None
        keys = tuple(getattr(cls, k) for k in type(cls).typekeys)
        if any([k is None for k in keys]):
            return None
        elif len(keys) is 0:
            return None
        elif len(keys) == 1:
            return keys[0]
        else:
            return keys

    @xprops.cachedproperty
    def abstract(cls):
        """True if any type parameter is None."""
        unfilled = any([c is None for c in cls.typeparameters])
        return cls.is_pending_archetype or unfilled

    @xprops.cachedproperty
    def is_pending_archetype(cls):
        """True if there are unprocessed metadescriptors in namespace."""
        return any([isinstance(getattr(cls, k, None), nxd.MetaDescriptor)
                    for k in dir(cls)])

    def subtype(cls, *traits, name=None, **kwargs):
        """Returns a subtype of cls with possible modifiers."""
        return nxtype(cls, *traits, name=name, **kwargs)


def nxtype(basetype, *traits, name=None, **kwargs):
    """Programatically returns a type over the supplied base type.

    params:
        traits: an iterable of traits to append to the new type.
        name: optional string, otherwise metaclass baptizing is used.
        kwargs: keyword arguments to be passed to the metaclass.
    """
    metatype = type(basetype)
    if traits:
        kwargs['traits'] = traits
    else:
        kwargs.setdefault('traits', None)
    name = fun.get(name, metatype.__baptize__(basetype, **kwargs))
    cls = types.new_class(name, (basetype,), kwds=kwargs)
    # Assigns the caller's module to the new class by default.
    try:
        cls.__module__ = getmodule(sys._getframe(1)).__name__
    except AttributeError:  # Interactive mode
        cls.__module__ = basetype.__module__
    return cls


def archetype(cls):
    """A class decorator that signals an archetype.

    Archetype stand out as types in that they are designed to form the root
    of so-called 'clades', or families of objects that all share the same basic
    structure. In terms of implementation, the archetype is simply a base
    class for derived types, but what makes the clade a particular
    relationship is the specific way in which those derived types relate
    to the archetype. Archetypes declare meta descriptors,  particularly
    type attributes / type parameters which are basically class variables that
    can be set programatically. An example could be a family of nested list
    types each with a set depth. The archetype is an abstract class defining
    recursive methods for dealing with nested lists, and the concrete types are
    defined by their depth, which is the clade's type parameter.
    The difference between type attributes and type parameters is that the
    former are essentially optional, whereas type parameters must have a
    non-None value for a type to be considered concrete. Furthermore, type
    parameters can only be set once at instantiation.
    * Clade members can also enrich the archetype with traits, as generally
    enabled by NxType.
    * When an archetype is declared, a metaclass is generated programatically
    to handle subclassing. The decorator outputs a new class that has
    the same name and attributes as the decorated class, but is an
    instance of the clade's metaclass, as well as an instance of the
    ArcheType base metaclass -see nxmeta.ArcheType for details.
    * Note that clade is simply a concept used to refer to objects sharing the
    same archetype, but there is no explicit implementation of a clade
    object. On the other hand, anaximander provide the convenience function
    clade, which returns an object's archetype. By convention, the clade
    of a type that derives from an archetype is also that archetype.
    """
    mcl = archmeta(cls)
    return mcl(cls.__name__, (cls,), {})
