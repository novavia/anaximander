#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch of the attrs package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from functools import partial

from attr import *

_attributes = attributes

# =============================================================================
# Patching
# =============================================================================


def attributes(maybe_cls=None, *, these=None, repr_ns=None, repr=True,
               cmp=True, hash=True, init=True, slots=False, frozen=False,
               str=False, inherit=True):
    """Patch of attr to add the *inherit* keyword argument.

    Additional anaximander documentation:
    params:
        inherit (bool): if False, then attributes declared in parent classes
            are ignored. The intended use is for the programmer to explicitly
            declare all attributes of a class, irrespective of its inheritance
            chain.
    """
    if maybe_cls is None:
        return partial(attributes, these=these, repr_ns=repr_ns,
                       repr=repr, cmp=cmp, hash=hash, init=init,
                       slots=slots, frozen=frozen, str=str,
                       inherit=inherit)
    attrs_attrs = {}
    # Terrible hack to counterharck attr's bad metaprogramming behavior
    # We strip all base classes of __attrs_attrs__ so they cannot be
    # collected.
    if inherit is False:
        for c in maybe_cls.__mro__[1:]:
            if '__attrs_attrs__' in c.__dict__:
                attrs_attrs[c] = c.__attrs_attrs__
                del c.__attrs_attrs__
    # Apply the original decorator.
    cls = _attributes(maybe_cls, these=these, repr_ns=repr_ns,
                      repr=repr, cmp=cmp, hash=hash, init=init,
                      slots=slots, frozen=frozen, str=str)
    # Then put the __attrs_attrs__ back in place.
    for c, d in attrs_attrs.items():
        setattr(c, '__attrs_attrs__', d)
    return cls


attributes.__doc__ = _attributes.__doc__ + attributes.__doc__

# Reassign shortcuts
s = attrs = attributes
