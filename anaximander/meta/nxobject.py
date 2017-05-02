#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the abstract base class NxObject.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from functools import partial

from .nxtype import NxType
from .nxmeta import MetaError
from ..registries.folios import RegistrableObject
from ..utilities import nxattr

_attributes = nxattr.attributes


__all__ = ['NxObject']

# =============================================================================
# NxType declaration
# =============================================================================


class NxObject(RegistrableObject, metaclass=NxType, registry=None):
    """The Anaximander base object class."""

    def __init__(self):
        type(self).__registry__.register(self)


def attributes(maybe_cls=None, *, these=None, repr_ns=None, repr=True,
               cmp=True, hash=True, init=True, slots=False, frozen=False,
               str=False, inherit=True, super=NxObject):
    """Patch of attr to ensure that __init__ is preserved on NxObjects.

    This patch automatically injects an inherited __init__ method into
    attr's post_init method -either by creating one, or by creating a composed
    method if an __attr_post_init__ already exists. In the latter case, the
    inherited __init__ is executed after the __init__ method created by
    attr, but before the declared __attr_post_init__. Note that if
    init is False, super is ignored and the behavior is the same as with
    the original attributes decorator.
    By default, super is set to NxObject and will hence execute NxObject's
    __init__ method. Other valid values are any type in the decorated class'
    method resolution order whose init method takes 'self' as its sole
    argument, and False or None to indicate that no inherited init method
    should be run. If the decorated class doesn't inherit from NxObject
    the functionality is still available, but the default with revert to
    False.
    """
    if maybe_cls is None:
        return partial(attributes, these=these, repr_ns=repr_ns,
                       repr=repr, cmp=cmp, hash=hash, init=init,
                       slots=slots, frozen=frozen, str=str,
                       inherit=inherit, super=super)
    if super is NxObject and not issubclass(maybe_cls, NxObject):
        super = False
    if init is True and super not in (False, None):
        if super not in maybe_cls.__mro__:
            msg = "'super' argument in attributes must be a super class " + \
                "of the decorated class."
            raise MetaError(msg)
        try:
            declared_post_init = maybe_cls.__attrs_post_init__
        except AttributeError:
            post_init = super.__init__
        else:
            def post_init(self):
                """Wraps a super __init__ and declared __attrs_post_init__."""
                super.__init__(self)
                declared_post_init(self)
        maybe_cls.__attrs_post_init__ = post_init
    # Apply the original decorator.
    cls = _attributes(maybe_cls, these=these, repr_ns=repr_ns,
                      repr=repr, cmp=cmp, hash=hash, init=init,
                      slots=slots, frozen=frozen, str=str, inherit=inherit)
    return cls


attributes.__doc__ = _attributes.__doc__

# Reassign shortcuts
nxattr.s = nxattr.attrs = nxattr.attributes = attributes
