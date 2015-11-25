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

import inspect

#==============================================================================
### Attributes handling
#==============================================================================

def spformat(collection, singular='item', plural=None):
    """Returns 'n {item}' or 'n {items}' where n is the collection's length.

    :param collection: a collection-like object (defines __len__).
    :param singular: the naming of an item, defaults to 'item'.
    :param plural: plural form, defaulting to singular + 's'.
    :return: formatted string.
    """
    plural = plural or singular + 's'
    n = len(collection)
    if n <= 1:
        return '{n} {sf}'.format(n = n, sf = singular)
    else:
        return '{n} {pf}'.format(n = n, pf = plural)


def dictionarize(*args):
    """Builds a dictionary from locally named variables.

    If a variable x is defined in the caller's frame, and 'x' is passed
    to dictionarize, the function returns {'x': x}.

    :param *args: a sequence of local variables names.
    :return: a dictionary of locally named variables.
    """
    caller_frame = inspect.currentframe().f_back
    return {a:caller_frame.f_locals[a] for a in args}


def get(val, default=None):
    """This function returns val if not None, else default."""
    return val if val is not None else default


def metargs(cls):
    """Returns the name, bases and attributes of the supplied class."""
    return (cls.__name__, cls.__bases__, cls.__dict__.copy())


