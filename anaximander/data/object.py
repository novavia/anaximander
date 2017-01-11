#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the DataObject base archetype.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from anaximander.utilities import xprops
from anaximander.meta.nxtype import archetype
from anaximander.meta.nxobject import NxObject

# =============================================================================
# DataObject class
# =============================================================================


@archetype
class DataObject(NxObject):
    """Archetype for all Data objects."""

    @xprops.weakproperty
    def context(self):
        """An optional context a Data object can refer to."""
        return None
