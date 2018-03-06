#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the Structure base class for object-like entities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from .meta import archetype
from .meta.nxobjects import NxObject

__all__ = ['Structure']

# =============================================================================
# Structure base class
# =============================================================================


@archetype
class Structure(NxObject):
    pass
