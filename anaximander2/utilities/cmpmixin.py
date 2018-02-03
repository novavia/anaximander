#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a Mixin class that provides rich comparison methods.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class ComparableMixin:
    """Mixin class to provide comparison methods.

    Usage requires that classes that implement this mixin define a method
    __cmpkey__, which is used as the basis for comparison between instances.
    Additionally, comparison checks the other item's type. The default
    is that the other item should be an instance of self's type. However
    it is possible to override that behavior by specifying an iterable
    of __cmptypes__ in the host class. Alternatively, __cmptypes__ can
    be set to None, in which case only strict type equality will return
    a successful comparison.
    An alternative to defining __cmpkey__ is to declare a __cmpattrs__
    iterable of strings. In this case, the comparison operator performs
    a tuple comparison by pulling the corresponding attributes from
    the objects to be compared.
    """

    def _compare(self, other, method):
        if hasattr(self, '__cmpkey__'):
            key = lambda inst: inst.__cmpkey__()
        elif hasattr(self, '__cmpattrs__'):
            key = lambda inst: tuple(getattr(inst, a)
                                     for a in self.__cmpattrs__)
        else:
            return NotImplemented
        try:
            keycheck = method(key(self), key(other))
        except (AttributeError, TypeError):
            # __cmpkey__ not implemented, or return different type.
            return NotImplemented
        cmptypes = getattr(self, '__cmptypes__', ())
        if cmptypes is None:
            typecheck = type(other) == type(self)
        else:
            cmptypes += (type(self),)
            typecheck = isinstance(other, cmptypes)
        if typecheck:
            return keycheck
        else:
            return NotImplemented

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)
