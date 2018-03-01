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

import threading
import weakref

from .functions import typecheck


# Global lock on reading properties
# Programmers are responsible for thread safety in setting properties
# However the module locks the cache for reading properties
LOCK = threading.RLock()

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

    @property
    def name(self):
        try:
            return self.fget.__name__
        except AttributeError:
            return '...'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # We prefer obj.__dict__ to getattr, as this enables cachedproperty
        # to operate on instances or types. With the latter, getattr
        # would look up the inheritance chain which is often not desirable
        # and somewhat contrary to the notion of a chached property.
        with LOCK:
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
        return type(self)(fget, self.fset, self.fdel, self.__doc__, self.cache)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__, self.cache)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__, self.cache)

    @staticmethod
    def reset(obj):
        """Resets all cached properties on obj."""
        cls = type(obj)
        for attr in dir(cls):
            v = getattr(cls, attr, None)
            if isinstance(v, cachedproperty):
                v.__delete__(obj)


class settablecachedproperty(cachedproperty):
    """Cached property with a simple setter.

    This property can also define a validator, which is called by
    the default setter. The validator takes an instance and a value as
    arguments and must return True if validation passes. The validator can
    implement its own error raising mechanism, otherwise a ValueError is
    raised.
    """

    def __init__(self, fget=None, fset=None, fdel=None, fval=None, doc=None,
                 cache=None):
        super().__init__(fget, fset, fdel, doc, cache)
        self.fval = fval

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
            return
        elif self.fval is not None:
            try:
                assert self.fval(obj, value)
            except Exception as e:
                msg = f"Invalid value {value} passed to property " + \
                      f"{self.name} of {obj}"
                etype = type(e)
                if etype is AssertionError:
                    etype = ValueError
                raise etype(msg)
        setattr(obj, self.cache, value)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.fval,
                          self.__doc__, self.cache)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.fval,
                          self.__doc__, self.cache)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.fval,
                          self.__doc__, self.cache)

    def validator(self, fval):
        return type(self)(self.fget, self.fset, self.fdel, fval,
                          self.__doc__, self.cache)


class singlesetproperty(settablecachedproperty):
    """Cached property that can only be set once.

    Note that deletion remains possible and that a new value can be passed
    to the cache following a del event. This is maintained for compatibility
    and consistency with the base class behavior.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        with LOCK:
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

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        with LOCK:
            try:
                return getattr(obj, self.cache)()
            except (AttributeError, TypeError):
                if self.fget is None:
                    return None
                return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
            return
        elif self.fval is not None:
            try:
                assert self.fval(obj, value)
            except Exception as e:
                msg = f"Invalid value {value} passed to property " + \
                      f"{self.name} of {obj}"
                etype = type(e)
                if etype is AssertionError:
                    etype = ValueError
                raise etype(msg)
        if value is not None:
            setattr(obj, self.cache, weakref.ref(value))
        else:
            setattr(obj, self.cache, None)


def typedweakproperty(*types):
    """A function helper to declare a weakproperty with type checking."""
    def decorator(fget):
        def fval(obj, value):
            if not typecheck(*types)(value):
                raise TypeError
            return True
        return weakproperty(fget, fval=fval)
    return decorator
