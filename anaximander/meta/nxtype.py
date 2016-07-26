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

#==============================================================================
### Import statements
#==============================================================================

import itertools
import types
from abc import ABCMeta

from . import nxobject, meta
from ..utilities import xprops
from ..utilities import functions as fun

#==============================================================================
### NxType declaration
#==============================================================================

class NxType(ABCMeta, nxobject, metaclass=meta, basename=None):
    """The Anaximander base metaclass.
    
    The basename for NxType is set to None in order to emphasize the
    abstract nature of NxType -i.e. it is not intended to directly
    produce any type, as this is left to concrete subclasses.    
    """
    __archetype__ = None #  The archetype upon which the metaclass is built.
    
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
        cls._type_folios = set()
        
    @property
    def _folios(cls):
        """This property is used to hold references to registration folios.
        
        Note that NxType instances, which are classes, also have a
        _folios property in their dictionary, which applies to objects.
        This design is made possible by using two different names for
        the underlying variable that stores the folios set ('_object_folios'
        for objects, '_type_folios' for types).        
        """
        return cls._type_folios


def nxtype(basetype, *traits, name=None, **kwargs):
    """Programatically returns a type over the supplied base type.
    
    :param traits: an iterable of traits to append to the new type.
    :param name: optional string, otherwise metaclass baptizing is used.
    """
    metatype = type(basetype)  # FIXME: not if basetype is an archetype!
    name = fun.get(name, metatype.__baptize__(*traits, **kwargs))
    return types.new_class(name, (basetype,))

