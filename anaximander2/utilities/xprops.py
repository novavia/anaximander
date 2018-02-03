#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides extended property descriptors.

The properties are cachedproperty, settablecachedproperty, weakproperty, and
classproperty.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import weakref

from .functions import typecheck

# =============================================================================
# Properties
# =============================================================================


class cachedproperty(property):
    """A lazily evaluated but cached property descriptor.

    cachedproperty implement __delete__, which in this particular context
    empties the cache so that the property can be recalculated on the next
    call.
    There is also a reset class method that can be passed an object
    instance: all cached propeties of the instance will then be reset,
    i.e. their cache is emptied.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None,
                 cache=None):
        super().__init__(fget, fset, fdel)
        if cache is None and fget is not None:
            self.cache = '_' + fget.__name__
        else:
            self.cache = cache
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # We prefer obj.__dict__ to getattr, as this enables cachedproperty
        # to operate on instances or types. With the latter, getattr
        # would look up the inheritance chain which is often not desirable
        # and somewhat contrary to the notion of a chached property.
        try:
            return obj.__dict__[self.cache]
        except KeyError:
            if self.fget is None:
                raise AttributeError("Unreadable attribute.")
            value = self.fget(obj)
            setattr(obj, self.cache, value)
            return value

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
        else:
            raise AttributeError("Can't set attribute.")

    def __delete__(self, obj):
        if self.fdel is not None:
            self.fdel(obj)
            return
        try:
            delattr(obj, self.cache)
        except AttributeError:
            pass

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel)

    @staticmethod
    def reset(obj):
        """Resets all cached properties on obj."""
        cls = type(obj)
        for attr in dir(cls):
            v = getattr(cls, attr, None)
            if isinstance(v, cachedproperty):
                v.__delete__(obj)


class settablecachedproperty(cachedproperty):
    """Cached property with a simple setter."""

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
        else:
            setattr(obj, self.cache, value)


class singlesetproperty(settablecachedproperty):
    """Cached property that can only be set once.

    Note that deletion remains possible and that a new value can be passed
    to the cache following a del event. This is maintained for compatibility
    and consistency with the base class behavior.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return obj.__dict__[self.cache]
        except KeyError:
            if self.fget is None:
                raise AttributeError("Unreadable attribute.")
            return self.fget(obj)

    def __set__(self, obj, value):
        if self.cache not in obj.__dict__:
            super().__set__(obj, value)
        else:
            raise AttributeError("Can't set attribute.")


class weakproperty(settablecachedproperty):
    """A cached property who stores a weak reference of a target attribute.

    weakproperty decorates a function that returns a default value -usually
    None. It is meant to be set and will keep a weak reference in cache.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None,
                 cache=None, types=None):
        """types is an optional iterable of valid types."""
        super().__init__(fget, fset, fdel, doc, cache)
        self.types = types

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return getattr(obj, self.cache)()
        except (AttributeError, TypeError):
            if self.fget is None:
                return None
            return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
        elif value is not None:
            if self.types is not None:
                if not typecheck(*self.types)(value):
                    raise TypeError
            setattr(obj, self.cache, weakref.ref(value))
        else:
            setattr(obj, self.cache, None)


def typedweakproperty(*types):
    """A function helper to declare a weakproperty with type checking."""
    def decorator(fget):
        return weakproperty(fget, types=types)
    return decorator
