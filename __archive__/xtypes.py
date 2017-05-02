#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended types to build stronger taxonomies.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Import statements
#==============================================================================

from collections import ChainMap
from inspect import getmodule
import sys
import types

from .. import nxtype, nxobject
from ..utilities import functions as fun
from ..utilities.functions import lmap

#==============================================================================
### Customized exception class
#==============================================================================


class MetaError(Exception):
    """Metaprogrammation exception class."""
    pass

#==============================================================================
### Assembler machinery.
#==============================================================================


class Assembler(nxobject):
    """Assemblers are callables that return types.

    The Assembler class as a whole is a type factory. Any Assembler
    instance can be made to return any kind of classes, similarly to
    the new_class function defined in the types standard library.
    Assembler combines this functionality with a behavior akin to
    partial in the functools standard library, such that Assembler
    instances can be built iteratively from multiple calls.
    Assembler can automtically name a class if the metaclass used to
    instantiate it features a classmethod __baptize__, which takes
    bases, patch and **keys as arguments and returns a name.
    """
    # Class defaults, which may be overriden by subclasses.
    bases = ()  # type bases.
    metaclass = None  # metaclass used to create types.
    patch = None  # optional namespace patch.
    slots = None  # Optional tuple of __slot__ strings.
    keys = None  # optional type keys, passed to the metaclass.

    def __init__(self, *bases, metaclass=None, patch=None, slots=None, **keys):
        """Initializes an Assembler. All arguments are optional.

        :param *bases: a sequence of base classes.
        :param metaclass: an optional metaclass.
        :param patch: a dictionary of additional class attributes & methods.
        :param slots: a tuple of __slots__ strings.
        :param **keys: type keys / metaclass keyword arguments, as applicable.
        :returns: an Assembler instance.
        """
        self.keys = {}
        kwargs = ChainMap(keys, lmap('metaclass', 'patch', 'slots'))
        self.update(*bases, **kwargs)

    def update(self, *bases, metaclass=None, patch=None, slots=None, **keys):
        """Updates an Assembler instance with new / additional parameters.

        Note that all of the parameters except keys update through overwriting.
        One cannot, say, add a base class or augment an existing patch with an
        additional method. For this, a new bases tuple or patch dictionary
        need to be created then passed to the update method.
        Keys work the other way round, with a dictionary update.
        """
        if bases:
            self.bases = bases
        if metaclass is not None:
            self.metaclass = metaclass
        if patch is not None:
            self.patch = patch
        if slots is not None:
            self.slots = slots
        self.keys.update(keys)

    def __call__(self, name=None, **kwargs):
        """The main interface, returns either a type or an updated Assembler.

        Admissible kwargs include bases, metaclass, patch and slots, which have
        the result of updating those instance variables locally to the call.
        Additional kwargs are treated as type keys updates.
        If the attempt to create a class from the supplied argument
        fails with a TypeError, an Assembler instance  with updated arguments
        is returned in lieu of a type.
        """
        bases = kwargs.pop('bases', self.bases)
        metaclass = kwargs.pop('metaclass', self.metaclass)
        patch = kwargs.pop('patch', self.patch)
        slots = kwargs.pop('slots', self.slots)
        keys = self.keys.copy()
        if slots is not None:
            keys['slots'] = slots
        keys.update(kwargs)
        metarg = {'metaclass': metaclass} if metaclass else {}
        kwds = ChainMap(metarg, keys)
        # Callable that returns function's return value if not a type.
        assembler = lambda: type(self)(*bases, patch=patch, **kwds)
        if not name:
            # Find the appropriate metaclass by calling prepare with mock name
            metaclass, _, _ = types.prepare_class('mock', bases, kwds)
            if not hasattr(metaclass, '__baptize__'):
                return assembler()
            name = metaclass.__baptize__(bases, **keys)
        if patch is not None:
            exec_body = lambda ns: ns.update(patch)
        else:
            exec_body = lambda ns: ns
        try:
            cls = types.new_class(name, bases, kwds, exec_body)
        # Failure to create new class, normally attributable to insufficient
        # attribute specifications.
        except MetaError:
            return assembler()
        cls.__module__ = getmodule(sys._getframe(1))
        return cls

#==============================================================================
### Extended type definitions
#==============================================================================


def metatype(cls):
    """Type decorator, enables the decorated type to generate other types."""

    def subtype(cls, *mixins, name=None, patch=None, **kwargs):
        """Returns a new type from provided arguments.

        :param *mixins: mixin bases for the new class in addition to cls
        :param name: name for the new class, defaults to programmatic naming
        :param patch: a mapping of additional attributes for the new class
        :param kwargs: class kwargs, which can include a 'metaclass'
        """
        bases = (cls,) + mixins
        name = fun.get(name, cls.baptize(*mixins, patch=patch, **kwargs))
        exec_body = lambda ns: ns.update(patch) if patch else ns
        sub = types.new_class(name, bases, kwargs, exec_body)
        sub.__module__ = cls.__module__
        return sub