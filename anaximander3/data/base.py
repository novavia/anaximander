#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines a base metaclass for data objects.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

import abc


__all__ = ['DataObjectType']

# =============================================================================
# Base type
# =============================================================================


class DataObjectType(abc.ABCMeta):
    _registry = dict()  # Type registry

    def __init__(cls, name, bases, namespace):
        if 'schema' in namespace:
            cls._registry[cls.schema] = cls

    def __getitem__(cls, schema):
        for stype in type(schema).__mro__:
            try:
                type_ = cls._registry[stype]
                assert issubclass(type_, cls)
            except KeyError:
                continue
            except AssertionError:
                raise KeyError
        raise KeyError
