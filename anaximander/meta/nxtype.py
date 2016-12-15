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

import itertools
import types

from .nxmeta import NxMeta, nxmeta
from ..utilities import functions as fun
from ..registries.folios import RegistrableType

# =============================================================================
# NxType declaration
# =============================================================================


class NxType(RegistrableType, metaclass=NxMeta, basename=None):
    """The Anaximander base metaclass.

    The basename for NxType is set to None in order to emphasize the
    abstract nature of NxType -i.e. it is not intended to directly
    produce any type, as this is left to concrete subclasses.
    """
    __archetype__ = None  # The archetype upon which the type is built.

    @classmethod
    def __baptize__(mcl, basename=None, *args, **kwargs):
        """Generates a name for a programatically genereated type instance.

        :param basename: optionally supplied basename. The metaclass's
        __basename__ attribute (set by  meta) is used if None.
        """
        basename = fun.get(basename, mcl.__basename__)
        # XXX: Silly placeholder, will change
        try:
            idx = next(mcl.__name_index__)
        except AttributeError:
            mcl.__name_index__ = itertools.count()
            idx = next(mcl.__name_index__)
        return basename + '_' + str(idx)

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)


def nxtype(basetype, *traits, name=None, **kwargs):
    """Programatically returns a type over the supplied base type.

    :param traits: an iterable of traits to append to the new type.
    :param name: optional string, otherwise metaclass baptizing is used.
    """
    metatype = type(basetype)  # FIXME: not if basetype is an archetype!
    name = fun.get(name, metatype.__baptize__(*traits, **kwargs))
    return types.new_class(name, (basetype,))


def archetype(cls):
    """A class decorator that signals an archetype.

    Archetype stand out as types in that they are designed to form the root
    of so-called 'clades', or families of classes that all share the basic
    structure. Here is how the relationship between an archetypical class
    and the members of its clade different from regular class inheritance:
    * Archetypes can define metacharacters, which are basically class
    variables for which we would like to set different values for different
    types. An example could be a family of nested list types each with
    a set depth.
    * Clade members can also enrich the archetype with traits. Anaximander
    traits are essentially a framework for importing the behavior of
    a class into another class by association or aggregation rather than
    through explicit inheritance. For instance, a trait can be produced from
    a Spatial class, and types that have a geometry attribute will be
    made to inherit from the Spatial trait, which will confer them additional
    behaviors, such as a bounding box property.
    * When an archetype is declared, a metaclass is generated programatically
    to handle subclassing. The archetype itself is not an instance of that
    metaclass, but the other members of the clade are.
    * Clade members are registered into a 'cladogram', which is a tree-like
    structure that enables fast retrieving of types from a set of
    metacharacteristics and traits.
    """
    mcl = nxmeta(cls)
    cls.__archetype__ = cls
    cls.__clade__ = mcl
    return cls
