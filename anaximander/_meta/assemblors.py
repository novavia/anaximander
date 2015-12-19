#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemblors are callable objects that return types.

Assemblors are used in Anaximander to produce classes programatically.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Import statements
#==============================================================================

import types

from .nxmeta import NxBaseType, NxBaseObject
from ..arche import nxobject

#==============================================================================
### Assemblor class
#==============================================================================


class Assemblor(nxobject):
    """Assemblors are callables that return types.

    The Assemblor class as a whole is a type factory. Any Assemblor
    instance can be made to return any kind of class, similarly to
    the new_class function defined in the types standard library.
    Assemblor combines this functionality with a behavior akin to
    partial in the functools standard library, such that Assemblor
    instances can be built recursively from multiple calls.
    """
    # Class defaults, which may be overriden by subclasses.
    bases = (NxBaseObject,)
    meta = NxBaseType
    name = None
    patch = None
    keys = None

    def __init__(self, *bases, meta=None, name=None, patch=None, **keys):
        """Initializes an Assemblor. All arguments are optional.

        :param *bases: a sequence of base classes.
        :param meta: an optional metaclass.
        :param name: an optional class name, as naming may be automated.
        :param patch: a dictionary of additional class attributes & methods.
        :param **keys: metaclass keyword arguments, as applicable.
        :returns: an Assemblor instance.
        """
        if bases:
            self.bases = bases
        if meta:
            self.meta = meta
        if name:
            self.name = name
        self.patch = patch or dict()
        self.keys = keys or dict()

    def __call__(self, **kwargs):
        """The main interface, returning either a new Assemblor or a class.

        Admissible kwargs include bases, meta, name, and patch, which
        have the result of updating those instance variables locally to the
        call. If an Assemblor is returned, then the new instance has
        updated attributes.
        """
        bases = kwargs.pop('bases', self.bases)
        meta = kwargs.pop('meta', self.meta)
        name = kwargs.pop('name', self.name)
        patch = kwargs.pop('patch', self.patch)
        keys = self.keys.copy()
        keys.update(kwargs)
        try:
            if not name:
                name = meta.__baptize__(bases, **keys)
            exec_body = lambda ns: ns.update(patch)
            types.new_class(name, bases, keys, exec_body)
        except TypeError:
            return type(self)(bases, meta, name, patch, **keys)
