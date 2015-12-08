#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides extended property descriptors.

The properties are cachedproperty, settablecachedproperty and weakproperty.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
# Import statements
#==============================================================================

import weakref

#==============================================================================
# Properties
#==============================================================================


class cachedproperty(property):
    """A lazily evaluated but cached property descriptor.

    cachedproperty implement __delete__, which in this particular context
    empties the cache so that the property can be recalculated on the next
    call.
    There is also a reset class method that can be passed an object
    instance: all cached propeties of the instance will then be reset,
    i.e. their cache is emptied.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        super(cachedproperty, self).__init__(fget, fset, fdel)
        self.cache = '_' + fget.__name__
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return getattr(obj, self.cache)
        except AttributeError:
            if self.fget is None:
                raise AttributeError("Unreadable attribute.")
            setattr(obj, self.cache, self.fget(obj))
            return getattr(obj, self.cache)

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

    @classmethod
    def reset(cls, obj):
        """Resets all cached properties on obj (if declared in its class)."""
        attrs = type(obj).__dict__
        props = [k for k, v in attrs.items() if isinstance(v, cls)]
        for k in props:
            delattr(obj, k)


class settablecachedproperty(cachedproperty):
    """Cached property with a simple setter."""

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
        else:
            setattr(obj, self.cache, value)


class weakproperty(settablecachedproperty):
    """A cached property who stores a weak reference of a target attribute.

    weakproperty decorates a function that returns a default value -usually
    None. It is meant to be set and will keep a weak reference in cache.
    """

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        try:
            return getattr(obj, self.cache)()
        except AttributeError:
            if self.fget is None:
                return None
            return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is not None:
            self.fset(obj, value)
        elif value is not None:
            setattr(obj, self.cache, weakref.ref(value))
