#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander nxmeta, the type interpreter.

It also defines the Assembler class, which is a type factory.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from collections import ChainMap, OrderedDict
from functools import partial
from inspect import getmodule
import sys
import types

from .. import nxtype, nxobject
from ..utilities.xprops import settablecachedproperty
from ..utilities.functions import lmap

#==============================================================================
### Abstract base classes
#==============================================================================


class MetaError(Exception):
    """Metaprogrammation exception class."""
    pass


class nxdescriptor(nxobject):
    """The base Anaximander descriptor class."""

    @settablecachedproperty
    def name(self):
        """Optional name set by nxmeta."""
        return None

    def __metainit__(self, cls):
        """Called by nxmeta in its __init__ method."""
        pass


class typemethod(classmethod, nxdescriptor):
    """A classmethod that provides implementation to a metaclass method.

    For instance, nxmeta defines __getitem__ to allow bracket notations
    on types. The actual implementation of __getitem__ is kept at the type
    level in the method __metagetitem__, which is a typemethod. Compared to
    a classmethod, a type method simply checks that there is a registered
    metaclass method that is targeted by it -e.g. __getitem__ in the proposed
    example. Of course such check is optional, and a classmethod decoration
    would work in most cases but the typemethod makes the intent clearer.
    Type method naming is automated to allow a bijection between type method
    names and target metaclass methods, as follows:
    * __meth__ in the metaclass corresponds to the __metameth__ typemethod.
    * meth in the metaclass corresponds to the _metameth typemethod.
    * _meth in the metaclass corresponds to the _meta_meth typemethod.
    Typemethod accepts an optional string argument that names the target
    metaclass method (e.g. @typemethod(target='__getitem__')), in which case
    the decorated method can have any name its programmer likes. The type will
    still make an assignment to the conventional name, which then becomes an
    alias to the decorated typemethod. However this allows more flexibility in
    naming methods.
    """

    def __new__(cls, func=None, *, target=None):
        if func is None:
            return partial(cls, target=target)
        return classmethod.__new__(cls, func)

    def __init__(self, func=None, *, target=None):
        super().__init__(func)
        if target is None:
            try:
                self.target = self._target(func)
            except Exception as e:
                msg = "Typemethods must follow naming convention or " + \
                      "explicitly target a metaclass method with the " + \
                      "keyword 'target' in the decorator."
                raise ValueError(msg) from e
        else:
            self.target = target

    @classmethod
    def _target(cls, func):
        """Retrieves the metaclass target name from a function's name."""
        fname = func.__name__
        if fname.startswith('__meta'):
            return fname.replace('__meta', '__')
        elif fname.startwith('_meta'):
            return fname.replace('_meta', '')
        else:
            raise ValueError

    @classmethod
    def _func(cls, target):
        """Retrieves the conventional function name from a target."""
        if target.startswith('__'):
            return '__meta' + target[2:]
        else:
            return '_meta' + target

    def __metainit__(self, cls):
        if not hasattr(type(cls), self.target):
            msg = "{0} has invalid metaclass method target."
            raise MetaError(msg.format(self))
        setattr(cls, self._func(self.target), self)


def get_slots(obj):
    """Returns the slots of a slotted object."""
    return tuple(getattr(obj, s) for s in obj.__slots__)


def set_slots(obj, args):
    """Fills the slots of a slotted object."""
    list(setattr(obj, s, a) for s, a in zip(obj.__slots__, args))


def del_slots(obj):
    """Sets slots to None."""
    list(setattr(obj, s, None) for s in obj.__slots__)


class nxmeta(nxtype):
    """The base type interpreter.

    nxmeta is not only the base class for NxType, which provides a base
    to library and application types, but also its metaclass, such that
    concrete types are instances of nxmeta.
    Unlike type, nxmeta accepts keyword arguments:
    * patch: a dictionary of attributes and functions appended to a class
    definition at runtime;
    * slots: a tuple of strings that is passed as __slots__ to the
    generated type;
    * Other keyword arguments are interpreted as meta attributes, i.e.
    attributes used to generate different versions of types, and are assigned
    as class variables in the type that is created. Note that this interface
    prohibits named arguments from being used as meta attributes. An alternate
    name for meta attributes is type keys.
    """
    __typecount__ = 0  # A class variable that keeps track of instance count.

    @classmethod
    def __baptize__(mcl, bases, patch=None, slots=None, **keys):
        """Returns a programmatic type name.

        This is a default implementation meant to be overwritten in
        specialized metaclasses.
        """
        return mcl.__name__ + '_' + str(mcl.__typecount__ + 1)

    @classmethod
    def __prepare__(mcl, name, bases, patch=None, slots=None, **keys):
        return OrderedDict()

    def __new__(mcl, name, bases, namespace, patch=None, slots=None, **keys):
        if patch is not None:
            namespace.update(patch)
        if slots is not None:
            namespace['__slots__'] = slots
        namespace.update(keys)
        namespace['__namespace__'] = namespace  # Keeps an OrderedDict
        return type.__new__(mcl, name, bases, namespace)

    def __init__(cls, name, bases, namespace, patch=None, slots=None, **keys):
        type.__init__(cls, name, bases, namespace)
        cls.__module__ = getmodule(sys._getframe(1))
        new_descriptors = cls._nxdescriptors(cls.__namespace__)
        for k, v in new_descriptors.items():
            v.name = k
            v.__metainit__(cls)
        if hasattr(cls, '__slots__'):
            # Adds the slots property.
            cls.slots = property(get_slots, set_slots, del_slots)
        if issubclass(cls, type):
            cls.__typecount__ = 0
        else:
            type(cls).__typecount__ += 1

    def __getitem__(cls, key):
        """Enables bracket calls on types."""
        return cls.__metagetitem__(key)

    def __metagetitem__(cls, key):
        """Default type __metagetitem__ implementation."""
        return NotImplemented

    @staticmethod
    def _nxdescriptors(namespace):
        """Extracts ordered nxdescriptors from a namespace."""
        nxd = lambda v: isinstance(v, nxdescriptor)
        return OrderedDict((k, v) for k, v in namespace.items() if nxd(v))

    @property
    def nxdescriptors(self):
        """Returns the set of nxdescriptors found in a type."""
        return set(self._nxdescriptors(self.__dict__).values())

    @classmethod
    def cm(mcl):
        print('meta class method')

    def im(cls):
        print(cls.__name__)


class Assembler(nxobject):
    """Assemblers are callables that return types.

    The Assembler class as a whole is a type factory. Any Assembler
    instance can be made to return any kind of classes, similarly to
    the new_class function defined in the types standard library.
    Assembler combines this functionality with a behavior akin to
    partial in the functools standard library, such that Assembler
    instances can be built recursively from multiple calls.
    Assembler can automtically name a class if the metaclass used to
    instantiate it features a classmethod __baptize__, which takes
    bases, patch and **keys as arguments and returns a name.
    """
    # Class defaults, which may be overriden by subclasses.
    bases = ()  # type bases.
    metaclass = None  # metaclass used to create types.
    patch = None  # optional dictionary to patch onto the namespace.
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
        if patch is not None:
            keys['patch'] = patch
        if slots is not None:
            keys['slots'] = slots
        keys.update(kwargs)
        metarg = {'metaclass': metaclass} if metaclass else {}
        kwds = ChainMap(metarg, keys)
        # Callable that returns function's return value if not a type.
        assembler = lambda: type(self)(*bases, **kwds)
        if not name:
            # Find the appropriate metaclass by calling prepare with mock name
            metaclass, _, _ = types.prepare_class('mock', bases, kwds)
            if not hasattr(metaclass, '__baptize__'):
                return assembler()
            name = metaclass.__baptize__(bases, **keys)
        try:
            cls = types.new_class(name, bases, kwds)
        # TODO: replace with prototype error
        except TypeError:
            return assembler()
        cls.__module__ = getmodule(sys._getframe(1))
        return cls
