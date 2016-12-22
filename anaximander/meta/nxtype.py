#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the abstract base metaclass NxType.

All Anaximander object classes are instances of NxType. However NxType
itself gets subclassed into more specialized metaclasses. NxType defines
the behaviors common to all Anaximander types. These include:
* Type registration. Types get registered by their metaclass, which is akin
to say that NxType and its derivatives register their instances. Because
Anaximander implements the notions of traits and metacharacters, type
registration is enabled by a specialized data structure called a Cladogram,
which provides an indexing facility for types.
* Descriptors interpretation. Anaximander implements specialized descriptors
and metadescriptors that are interpreted by NxType at runtime in order
to set the behaviors of new types -as well as of new metaclasses that may be
created programatically, which is where metadescriptors come in.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc
from inspect import getmodule
import sys
import types

from .nxmeta import NxMeta, archmeta, protometa
from ..utilities import functions as fun
from ..utilities import xprops
from ..registries.folios import RegistrableType

# =============================================================================
# NxType declaration
# =============================================================================


class MetaDescriptor(abc.ABC):
    """Base class for metadescriptors.

    Metadescriptors are intended to be inserted in type declarations
    to modify behavior for derived types. Their scope is types rather than
    objects, and thus they are declarative devices that get processed to
    become descriptors in a metaclass, hence the name metadescriptor.
    The NxType base metaclass systematically collects metadescriptors found
    in type declarations, strips them from the type's namespace, and park
    them into a __metadescriptors__ dictionary. For regular types this
    accomplishes nothing, but if a type is decorated with @archetype or
    @prototype, then the __metadescriptors__ dictionary is interpreted in
    order to create a new metaclass that implements the behaviors programmed
    in the metadescriptors.
    Metadescriptor behavior is bound to a metaclass in two stages. In the
    first stage, NxType passes the declaring type to each metadescriptor.
    In the second stage, the __call__ method of the metadescriptor instance
    is called and supplied a metaclass whose dictionary gets modified as
    a result.
    """

    @xprops.singlesetproperty
    def cls(self):
        """The declaring class, set by NxType."""
        return None

    @xprops.singlesetproperty
    def name(self):
        """The declared name, set by NxType."""
        return None

    @abc.abstractmethod
    def __call__(self, mcl):
        """Adds targeted behavior to the supplied metaclass."""
        if self.cls is None or self.name is None:
            msg = "Cannot call unbound metadescriptor instance."
            raise TypeError(msg)


class NxType(abc.ABCMeta, RegistrableType, metaclass=NxMeta, basename=''):
    """The Anaximander base metaclass.

    The basename for NxType is set to None in order to emphasize the
    abstract nature of NxType -i.e. it is not intended to directly
    produce any type, as this is left to concrete subclasses.
    NxType forbids multiple inheritance, instead allowing to compose
    types from a basetype and any number of traits.
    Anaximander traits are essentially a framework for importing the behavior
    of a class into another class by association or aggregation rather than
    through explicit inheritance. For instance, a trait can be produced from
    a Spatial class, and types that have a geometry attribute will implement
    the Spatial trait, which confers them additional behaviors, such as a
    bounding box property.
    """
    __archetype__ = None  # The archetype upon which the type is built.

    @classmethod
    def __baptize__(mcl, basetype, *traits, **kwargs):
        """Generates a name for a programatically generated type instance."""
        idx = next(mcl.__counter__)
        return mcl.__basename__ + '_' + str(idx)

    def __new__(mcl, name, bases, namespace, traits=None, **kwargs):
        """Collects metadescriptors and creates new NxType."""
        if not len(bases) == 1:
            raise TypeError("Anaximander types admit exactly one base class.")
        metadescriptors = {}
        for k, v in dict(namespace).items():
            if isinstance(v, MetaDescriptor):
                metadescriptors[k] = v
                del namespace[k]
        namespace['__metadescriptors__'] = metadescriptors
        # Note: this is ABCMeta.__new__
        return super().__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, traits=None, **kwargs):
        # Note: this is RegistrableType.__new__
        super().__init__(name, bases, namespace)
        # Bind the metadescriptors to the type that declared them.
        for name, md in cls.__metadescriptors__.items():
            md.cls = cls
            md.name = name

    def subtype(cls, *traits, name=None, **kwargs):
        """Returns a subtype, equivalent to nxtype."""
        return nxtype(cls, *traits, name=name, **kwargs)


def archetype(cls):
    """A class decorator that signals an archetype.

    Archetype stand out as types in that they are designed to form the root
    of so-called 'clades', or families of objects that all share the basic
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


def prototype(cls):
    """Decorates a class to be a prototype, i.e. an abstract archetype."""
    mcl = protometa(cls)
    return mcl(cls.__name__, (cls,), {})


def nxtype(basetype, *traits, name=None, **kwargs):
    """Programatically returns a type over the supplied base type.

    params:
        traits: an iterable of traits to append to the new type.
        name: optional string, otherwise metaclass baptizing is used.
        kwargs: keyword arguments to be passed to the metaclass.
    """
    metatype = type(basetype)
    name = fun.get(name, metatype.__baptize__(basetype, *traits, **kwargs))
    cls = types.new_class(name, (basetype,))
    cls.__module__ = getmodule(sys._getframe(1))
    return cls
