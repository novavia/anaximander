#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of general-purpose utility functions.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import sys

#==============================================================================
### Attributes handling
#==============================================================================


def get(val, default=None):
    """This function returns val if not None, else default."""
    return val if val is not None else default


def lmap(*args):
    """Builds a dictionary from locally named variables.

    If a variable x is defined in the caller's frame, and x is passed
    to lmap, the function returns {'x': x}.

    :param *args: a sequence of local variables names.
    :return: a dictionary of locally named variables.
    """
    caller_locals = sys._getframe(1).f_locals
    return {k: v for k, v in caller_locals.items() if v in args}

#==============================================================================
### String formatting
#==============================================================================


def spformat(obj, singular='item', plural=None):
    """Returns 'n {item}' or 'n {items}' where n is a collection's length.

    :param obj: a collection-like object (defines __len__) or an integer.
    :param singular: the naming of an item, defaults to 'item'.
    :param plural: plural form, defaulting to singular + 's'.
    :return: formatted string.
    """
    plural = plural or singular + 's'
    n = obj if isinstance(obj, int) else len(obj)
    if n <= 1:
        return '{n} {sf}'.format(n=n, sf=singular)
    else:
        return '{n} {pf}'.format(n=n, pf=plural)


class curlydict(dict):
    """A dictionary that returns '{k}' when passed a missing string key k."""

    def __missing__(self, key):
        return '{' + key + '}'


def lformat(string):
    """Applies format to string from the caller's local dictionary.

    For instance, if x is defined and equal to 3,
    lformat('x is {x}') returns 'x is 3'.
    """
    caller_locals = sys._getframe(1).f_locals
    return string.format(**curlydict(caller_locals))

#==============================================================================
### Metaprogramming
#==============================================================================


def metargs(cls):
    """Returns the name, bases and namespace of the supplied class."""
    return (cls.__name__, cls.__bases__, cls.__dict__.copy())


def ducktype(cls, mixin, *exclusions):
    """Adds attributes and methods from mixin to cls.

    Attributes to exclude can be passed to exclusions as strings.
    """
    xattrs = set(['__dict__', '__doc__', '__module__', '__qualname__',
                  '__weakref__', '_folios', '_cls_folios'])
    xattrs.update(exclusions)
    for name, attr in mixin.__dict__.items():
        if name not in xattrs:
            setattr(cls, name, attr)
