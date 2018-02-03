#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines descriptors used throughout the anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc

from anaximander2.utilities import xprops, functions as fun, cmpmixin as cmp

from . import NxMetaError

__all__ = ['NxDescriptor', 'TypeProperty', 'MetaCharacter']

# =============================================================================
# Base class
# =============================================================================


class NxDescriptor(cmp.ComparableMixin):
    """A basic data descriptor with registration mechanism.

    The descriptor accepts a one-time set name and class reference.
    The intended usage pattern is that the descriptor is declared
    in a class' namespace, and a metaclass collects the descriptors
    and assign them their name and the class in which they were
    declared. However it is possible to set those attributes at
    instantiation.
    The class reference is primarily useful for tracing and debugging
    purposes.
    NxDescriptor also features a registration mechanism, as practice
    demonstrates that it is often useful to enumerate the descriptors
    of a given type, whether handling an instance or a class.
    """
    __cmpattrs__ = ('name', 'cls')

    def __init__(self, name=None, cls=None):
        if name is not None:
            self.name = name
        if cls is not None:
            self.cls = cls

    @xprops.singlesetproperty
    def cls(self):
        """This property is intended to hold the declaring class."""
        return None

    @xprops.singlesetproperty
    def name(self):
        """This property is intended to hold the descriptor name."""
        return None

    def register(self, cls, registry_name='__nxdescriptors__'):
        """Registers self with the supplied class.

        Params:
            cls: A class in which to register the descriptor.
            registry_name: The attribute name of the class registry.
        """
        try:
            registry = getattr(cls, registry_name)
            registry[self.name] = self
        except (AttributeError, TypeError):
            msg = "NxDescriptors registration requires a valid class registry."
            raise NxMetaError(msg)

    @abc.abstractmethod
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return None

    def __set__(self, obj, value):
        raise AttributeError("Can't set attribute.")

    def __delete__(self, obj):
        raise AttributeError("Can't delete attribute.")

    def __repr__(self):
        return fun.iformat('name')(self)


class TypeProperty(NxDescriptor):
    """A descriptor that reads a property of the type.

    This requires the type's metaclass to declare its own data descriptor
    (otherwise the return value will always be the descriptor itself).
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(type(obj), self.name)


class MetaCharacter(NxDescriptor):
    """Metacharacters are metaclass descriptors for parametric types.

    Behavior-wise, metacharacters operate like a single-set cached property.
    The only difference is that an error will be raised if the caller
    object is not a type, which is a way to enforce that metacharacters
    are reserved to metaclasses.
    Nonetheless, anaximander types can declare metacharacters, which
    is a convenient shortcut to avoid having to deal with metaclasses or
    meta options when building simple archetypes. In that case, metacharacters
    are collected along with other nxdescriptors. When a type is decorated
    with the archetype decorator, its metacharacters are stripped and
    appropriated by the archetype's metaclass, then replaced with
    typeproperties.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        elif not isinstance(obj, type):
            msg = "Metacharacters are properties of types."
            raise TypeError(msg)
        try:
            return getattr(obj, '_' + self.name)
        except AttributeError:
            return None

    def __set__(self, obj, value):
        extant = self.__get__(obj)
        if extant is None:
            setattr(obj, '_' + self.name, value)
        else:
            msg = "Metacharacter values cannot be overriden."
            raise AttributeError(msg)
