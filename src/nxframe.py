#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander base framework classes.

The framework classes are constitutive of the framework. The framework layer
differs from the base layer, which provides a number of useful libraries to
program applications, and is itself implemented with the framework layer.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import types
from itertools import chain

import nxmeta
import nxregistries as nxr

#==============================================================================
### Abstract base classes
#==============================================================================


class NxFrameworkType(nxmeta.NxType):
    """Parent class to types in Anaximander's framework layer."""
    pass


class NxFrameworkObject(nxmeta.NxObject, metaclass=NxFrameworkType):
    """Parent class to Anaximander's framework layer objects."""
    pass

#==============================================================================
### Templating infrastructure
#==============================================================================


class Template(NxFrameworkType):
    """A specialized type whose instances are class Templates.

    A Template takes an argument key and optional argument _metaclass.
    The key is either a string or a tuple of strings that provide attribute
    names with which actual classes are created from the Template.
    For instance, a NestedList Template could be used to create NestedLists
    of set depths, and 'depth' would be the key (expected to be an integer)
    used to either set or retrieve a NestedList type of given depth. If the key
    is a tuple, then it is composite, and a tuple of parameters is required to
    create or retrieve a class. Key is made optional only in order to
    handle simple Template inheritance.
    _metaclass is the metaclass used to derive concrete types from a Template.
    A Template's metaclass is always Template, but because types created from
    a Template don't inherit from it, they can -and often need to, use a
    metaclass of their own. If not supplied, _metaclass defaults to
    type(t.__mro__[0]) where t is the template.
    Templates register the types that they generate in an NxTree, using the key
    -either simple or composite. As a result, it is possible to access types
    from the Template by using bracket notations, e.g. NestedList[3] will
    return a nested list type of depth 3.
    A class Template is written just like a regular class, and types derived
    from it will own all the attributes and methods defined in the class
    Template. However the Template metaclass alters the behavior of the
    Template class, which is not meant to be instantiated. As a result,
    __new__ is rewritten to generally require a concrete key instance from
    which the proper type can be retrieved. This means that it is possible
    to create a NestedList[3] object by calling
    NestedList(iterable, __depth__=3).
    Alternatively, the Template can defined a __type__ method that is
    called by __new__. One possible application is to determine the type
    dynamically from the parameters passed to the instantation interface.
    Template supports inheritance on a strictly restricted basis, i.e. only
    simple inheritance with no redefinition of the key or metaclass is
    guaranteed to work. Hence it is possible to create a SortedNestedList
    Template, and SortedNestedList[3] will be a subclass of NestedList[3].
    However anything more complicated than that such as multiple inheritance
    will probably fail. Also, while implementing a custom __new__ in a
    Template to be used by derived types can be made to work it can
    create some tricky situations. At any rate if that is done, the use
    of super to implement inheritance within __new__ should use the complete
    form, i.e. super(cls, cls).__new__(cls, *args, **kwargs). Or it might
    be easier to use an explicit class pointer instead of super.
    """

    @classmethod
    def __prepare__(mcl, name, bases, key=None, _metaclass=None):
        return super().__prepare__(mcl, name, bases)

    def __new__(mcl, name, bases, namespace, key=None, _metaclass=None):
        return super().__new__(mcl, name, bases, namespace)

    def __init__(tpl, name, bases, namespace, key=None, _metaclass=None):
        super().__init__(name, bases, namespace)
        if hasattr(tpl, '__namespace__'):  # Inherits from a parent Template.
            tpl.__namespace__.update(namespace)
            tpl.__original__ = False
        else:  # Original Template.
            tpl.__namespace__ = namespace
            tpl.__namespace__['__new__'] = tpl.__new__
            tpl.__key__ = key
            tpl.__metaclass__ = _metaclass or type(tpl.__mro__[1])
            tpl.__original__ = True
        tpl.__namespace__.pop('__qualname__', None)
        tpl.__namespace__.pop('__type__', None)
        tpl.__types__ = nxr.NxTree()

        def simple_new(tpl, *args, **kwargs):
            """Customized __new__ for templates."""
            try:
                key = kwargs.pop(tpl.__key__)
            except KeyError:
                if hasattr(tpl, '__type__'):
                    cls = tpl.__type__(*args, **kwargs)
                else:
                    raise
            else:
                cls = tpl.__types__.fetch(key)
            return cls(*args, **kwargs)

        def composite_new(tpl, *args, **kwargs):
            """Customized __new__ for templates."""
            try:
                key = (kwargs.pop(k) for k in tpl.__key__)
            except KeyError:
                if hasattr(tpl, '__type__'):
                    cls = tpl.__type__(*args, **kwargs)
                else:
                    raise
            else:
                cls = tpl.__types__.fetch(*key)
            return cls(*args, **kwargs)

        if isinstance(tpl.__key__, (tuple, list)):
            tpl.__new__ = staticmethod(composite_new)
        else:
            tpl.__new__ = staticmethod(simple_new)

    def __assemble__(tpl, *key):
        """Creates and registers a new class from template."""
        name = '_'.join(chain((tpl.__name__,), (str(k) for k in key)))
        kwds = {'metaclass': tpl.__metaclass__}
        if tpl.__original__:
            bases = tpl.__bases__
        else:  # We assume a single base that is another Template
            bases = (tpl.__bases__[0].__types__.fetch(*key),)

        def namespace(ns):
            ns.update(tpl.__namespace__)

        cls = types.new_class(name, bases, kwds, namespace)
        cls.__template__ = tpl
        if isinstance(tpl.__key__, (tuple, list)):
            for k, v in zip(tpl.__key__, key):
                setattr(cls, k, v)
        else:
            setattr(cls, tpl.__key__, key[0])
        tpl.__types__.register(cls, *key)
        return cls

    def __getitem__(tpl, key):
        if isinstance(key, tuple):
            try:
                return tpl.__types__.fetch(*key)
            except KeyError:
                return tpl.__assemble__(*key)
        else:
            try:
                return tpl.__types__.fetch(key)
            except KeyError:
                return tpl.__assemble__(key)

    def __repr__(tpl):
        return '<template' + type.__repr__(tpl)[6:]
