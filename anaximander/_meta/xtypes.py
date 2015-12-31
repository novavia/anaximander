#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended types to build stronger taxonomies.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Import statements
#==============================================================================

from .. import nxtype
from ..utilities import functions as fun

#==============================================================================
### Extended type definitions
#==============================================================================


class metatype(nxtype):
    """A type that can assemble other types."""
    _metacount = 0  # private class variable that incremented at instantiation.

#    @classmethod
#    def __prepare__(mcl, name, bases, **kwargs):
#        return {}
#
#    def __new__(mcl, name, bases, namespace, **kwargs):
#        return super().__new__(mcl, name, bases, namespace)
#
#    def __init__(cls, name, bases, namespace, **kwargs):
#        super().__init__(cls, name, bases, namespace)

    def __subclasses__(*args, **kwargs):
        try:
            return type.__subclasses__(*args, **kwargs)
        except TypeError:
            return type.__subclasses__(metatype)

    @classmethod
    def baptize(mcl, bases, patch=None, **kwargs):
        """Returns a synthetic names from type arguments."""
        return mcl.__name__ + '_' + str(mcl._metacount)

    def subtype(cls, *mixins, name=None, patch=None, **kwargs):
        """Returns a new type from provided arguments.

        :param *mixins: mixin bases for the new class in addition to cls
        :param name: name for the new class, defaults to programmatic naming
        :param patch: a mapping of additional attributes for the new class
        :param kwargs: class kwargs, which can include a 'metaclass'
        """
        bases = (cls,) + mixins
        name = fun.get(name, cls.baptize(*mixins, patch=patch, **kwargs))
        exec_body = lambda ns: ns.update(patch) if patch else ns
        sub = types.new_class(name, bases, kwargs, exec_body)
        sub.__module__ = cls.__module__
        return sub

import pdb; pdb.set_trace()
metatype.register(nxtype)
