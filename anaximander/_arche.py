#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the apeiron and abstract base classes nxtype and nxobject.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Import statements
#==============================================================================

from abc import ABCMeta

from .utilities import xprops

#==============================================================================
### ABC declaration
#==============================================================================

apeiron = type  # The unbounded origin of all concepts and things.


class nxtype(ABCMeta, apeiron):
    """Base type for all Anaximander types."""
    pass


class nxobject(metaclass=nxtype):
    """Base object for all Anaximander entities (Objects & Types)."""

    @xprops.cachedproperty
    def _folios(self):
        """This property is used to hold references to registration folios."""
        return set()
