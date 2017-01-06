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
from collections import OrderedDict
from inspect import getmodule
import sys
import types

from .metadescriptors import MetaDescriptor, TypeAttribute
from .nxmeta import NxMeta, archmeta, protometa, MetaError, ProtoType
from ..utilities import functions as fun
from ..registries.folios import RegistrableType

# =============================================================================
# NxType declaration
# =============================================================================


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
    def __baptize__(mcl, basetype, traits=None, **kwargs):
        """Generates a name for a programatically generated type instance."""
        idx = next(mcl.__counter__)
        return mcl.__basename__ + '_' + str(idx)

    # Note: this is superfluous in py3.6+
    @classmethod
    def __prepare__(mcl, name, bases, **kwargs):
        return OrderedDict()

    def __new__(mcl, name, bases, namespace, traits=None, **kwargs):
        """Collects metadescriptors and creates new NxType."""
        # Check for a single base
        if not len(bases) == 1:
            raise TypeError("Anaximander types admit exactly one base class.")

        # Collect and remove metadescriptors
        metadescriptors = OrderedDict()
        for k, v in OrderedDict(namespace).items():
            if isinstance(v, MetaDescriptor):
                metadescriptors[k] = v
                del namespace[k]
        namespace['__metadeclarations__'] = metadescriptors

        # Assigns type attributes, which may be declared in either the
        # namespace itself or the kwargs. If there is a conflict, the
        # namespace declarations take precedence.
        TypeAttribute.update(mcl, namespace, **kwargs)

        # Special override for subtypes of archetypes
        archetype = mcl.__archetype__
        if archetype is not None:
            base = bases[0]
            # If the base is the archetype, we replace it with __basetype__
            basetype = archetype.__basetype__
            if base is archetype:
                base = basetype
            elif not issubclass(base, basetype):
                msg = "Incorrect use of an ArcheType subclass."
                raise MetaError(msg)
            bases = (base,)

        # Note: this is ABCMeta.__new__
        cls = super().__new__(mcl, name, bases, namespace)
        archetype = type(cls).__archetype__
        if archetype is not None:
            for method in archetype.__newtypemethods__:
                cls = getattr(cls, method)()
        return cls

    def __init__(cls, name, bases, namespace, traits=None, **kwargs):
        # Note: this is RegistrableType.__new__
        super().__init__(name, bases, namespace)
        # Bind the metadescriptors to the type that declared them.
        for name, md in cls.__metadeclarations__.items():
            md.cls = cls
            md.name = name
        # If the metaclass has an __archetype__, then it attempts to
        # register the class with that archetype.
        archetype = type(cls).__archetype__
        if archetype is not None:
            archetype.nxregister(cls, **kwargs)
            # Execute typeinits in sequence
            for method in archetype.__typeinitmethods__:
                getattr(cls, method)()

    def subtype(cls, *traits, name=None, **kwargs):
        """Returns a subtype, equivalent to nxtype."""
        return nxtype(cls, *traits, name=name, **kwargs)

    @property
    def archetype(cls):
        return cls.__archetype__

    @property
    def prototype(cls):
        if isinstance(cls.__archetype__, ProtoType):
            return cls.__archetype__
        else:
            return None

    @property
    def typeattributes(cls):
        """Returns a tuple of type attributes, per __typeattributes__ spec."""
        return tuple(getattr(cls, n) for n in cls.__typeattributes__)

    @property
    def metacharacters(cls):
        """Returns a tuple of meta characters, per __metacharacters__ spec."""
        return tuple(getattr(cls, n) for n in cls.__metacharacters__)


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
    cls.__module__ = getmodule(sys._getframe(1)).__name__
    return cls

# =============================================================================
# archetype / prototype decorators
# =============================================================================


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


def clade(obj_or_type):
    """Retuns the archetype that defines the object's clade, if any.

    params:
        obj_or_type: any object or type
    returns:
        The object's archetype if it exists. By extension, the clade of
        a type is its base archetype if it has one. In all other cases,
        the function simply returns None.
    """
    try:
        return obj_or_type.__archetype__
    except AttributeError:
        return None
