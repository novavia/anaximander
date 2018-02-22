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
from collections import OrderedDict, ChainMap
from inspect import getmodule
import sys
import types

from ..utilities import functions as fun, xprops
from . import NxMetaError
from . import nxdescriptors as nxd
from .nxmetas import NxMeta, archmeta

__all__ = ['NxType', 'nxtype', 'archetype']

# =============================================================================
# NxMeta metaclass type
# =============================================================================


class NxType(abc.ABCMeta, metaclass=NxMeta):
    __basename__ = 'Nx'  # The basename, usable for metaclass instances.
    __archetype__ = None  # The archetype upon which the type is built.

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
            bases = (base,)
        # Set proto-class
        cls = super().__new__(mcl, name, bases, namespace)
        # Set the type descriptors
        # They are either supplied by the namespace, kwargs or inheritance
        attrs = ChainMap(namespace, kwargs)
        for k, v in mcl.__typedescriptors__.items():
            if k in attrs:
                setattr(cls, k, attrs[k])
            else:
                setattr(cls, v.cache, getattr(cls, v.cache, None))
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        # Processes nxdescriptors declared in the namespace
        nxd.MetaDescriptor.collect(cls, namespace)
        nxd.ObjectDescriptor.collect(cls, namespace)
        archetype = cls.__archetype__
        overtype = kwargs.get('overtype', False)
        if archetype is not None:
            if not any(c is None for c in cls.metacharacters):
                overtype = kwargs.get('overtype', False)
                archetype.nxregister(cls, overtype=overtype)
            else:
                cls.__new__ = classmethod(_abstract_new)

    def metadescriptors(cls, *types):
        """Returns a mapping of metadescriptors filtered by types.

        if no types are supplied, then all registered metadescriptors are
        returned.
        """
        if not types:
            types = (nxd.MetaDescriptor,)
        descriptors = fun.vfilter(fun.typecheck(*types),
                                  cls.__metadescriptors__)
        return OrderedDict(descriptors)

    def descriptors(cls, *types):
        """Returns a mapping of instance descriptors filtered by types.

        if no types are supplied, then all registered descriptors are
        returned.
        """
        if not types:
            types = (nxd.ObjectDescriptor,)
        descriptors = fun.vfilter(fun.typecheck(*types),
                                  cls.__objectdescriptors__)
        return OrderedDict(descriptors)

    @xprops.cachedproperty
    def metacharacters(cls):
        """Tuple of type properties for the archetype's metacharacters."""
        return tuple(getattr(cls, k) for k in type(cls).__metacharacters__)


def _abstract_new(cls, *args, **kwargs):
    """Implementation of __new__ for abstract archetypes and derivatives.

    This function is the __new__ method for any archetype or derivative
    whose metacharacters have any None values.
    """
    metacharacters = OrderedDict((k, getattr(cls, k))
                                 for k in type(cls).__metacharacters__)
    try:
        for k in list(metacharacters):
            if metacharacters[k] is None:
                metacharacters[k] = kwargs.pop(k)
    except KeyError:
        raise TypeError("Cannot instantiate a type without a full " +
                        "set of metacharacters.")
    return cls[tuple(metacharacters)](*args[1:], **kwargs)


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
    to the archetype. Archetypes define metacharacters, which are basically
    class variables for which we would like to set different values for
    different types. An example could be a family of nested list types each
    with a set depth. The archetype is an abstract class defining recursive
    methods for dealing with nested lists, and the concrete types are
    defined by their depth, which is the clade's metacharacter.
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
