#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the apeiron and the abstract base class nxobject.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Import statements
#==============================================================================

import xprops

#==============================================================================
### ABC declaration
#==============================================================================

apeiron = type  # The unbounded origin of all concepts and things.


class nxobject(object):
    """The base class of all Anaximander entitites (meta, type & object)."""

    @xprops.cachedproperty
    def _folios(self):
        """This property is used to hold references to registration folios."""
        return set()
