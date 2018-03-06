#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines NxObject, the root Anaximander object.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from collections import OrderedDict

from ..utilities import functions as fun
from .nxtypes import NxType
from . import NxMetaError


__all__ = ['NxObject']

# =============================================================================
# NxMeta metaclass type
# =============================================================================


class NxObject(metaclass=NxType):

    def __new__(cls, *args, **kwargs):
        if cls.abstract:
            if cls.is_pending_archetype:
                msg = f"Cannot instantiate pending archetype {cls}."
                raise NxMetaError(msg)
            typeparameters = OrderedDict((k, getattr(cls, k))
                                         for k in type(cls).typeparameters)
            try:
                for k in list(typeparameters):
                    if typeparameters[k] is None:
                        typeparameters[k] = kwargs.pop(k)
            except KeyError:
                raise NxMetaError("Cannot instantiate an object without a " +
                                  "full set of type parameters.")
            key = tuple(typeparameters[k] for k in type(cls).typekeys)
            klass = cls.archetype[key]
            nonkeyparams = OrderedDict([(k, typeparameters[k])
                                        for k in type(cls).nonkeyparameters])
            if nonkeyparams:
                klass = klass.subtype(**nonkeyparams)
            return klass(*args, **kwargs)
        else:
            return super().__new__(cls)

    def __repr__(self):
        return fun.iformat()(self)
