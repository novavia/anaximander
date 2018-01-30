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

# =============================================================================
# Adding a subclass validator
# =============================================================================


@attributes(repr=False, slots=True)
class _SubclassOfValidator(object):
    type = attr()

    def __call__(self, inst, attr, value):
        if not issubclass(value, self.type):
            msg = "'{name}' must be subclass of {type!r}"
            msg = msg.format(name=attr.name, type=self.type)
            raise TypeError(msg)

    def __repr__(self):
        return ("<subclass_of validator for type {t!r}>".format(t=self.type))


def _subclass_of(type):
    """
    A validator that raises a :exc:`TypeError` if the initializer is called
    with a wrong type for this particular attribute (checks are perfomed using
    :func:`issubclass` therefore it's also valid to pass a tuple of types).

    :param type: The type to check for.
    :type type: type or tuple of types

    The :exc:`TypeError` is raised with a human readable error message, the
    attribute (of type :class:`attr.Attribute`) and the expected type.
    """
    return _SubclassOfValidator(type)

validators.subclass_of = _subclass_of
