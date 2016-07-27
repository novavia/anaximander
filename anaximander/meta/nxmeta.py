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

from abc import ABCMeta

#==============================================================================
### NxMeta metaclass type
#==============================================================================

class NxMeta(ABCMeta):
    """A custom type for Anaximander metaclasses.

    Gotcha: the instances of this class are... metaclasses! This is pushing
    the abstraction of the model to a very unusual extreme and may make it
    difficult to understand right away. However it is worth noting, and
    probably helpful conceptually, that the methods are written using
    the name 'mcl' to refer to instances. For class methods, the designation
    'met' is used to refer to NxMeta itself.
    Note: this may need to implement a type registry in each newly
    created metatype, i.e. a cladogram.
    """

    def __new__(met, name, bases, namespace, basename):
        return type.__new__(met, name, bases, namespace)

    def __init__(mcl, name, bases, namespace, basename):
        """Initializes a new metaclass.
        
        :param basename: base name for types the new metaclass will create.
        """
        mcl.__basename__ = basename


def nxmeta(archetype):
    """A metaclass constructor from a supplied archetype.
    
    nxmeta creates a metaclass by subclassing the archetype's metaclass
    and modifiying its behavior based on any metadescriptors found
    in the archetype's declarations.
    """
    name = archetype.__name__ + 'Type'
    bases = (type(archetype),)
    basename = archetype.__name__
    return NxMeta(name, bases, {}, basename)
    