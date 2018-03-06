#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the Operator base class for model-level functions.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from .meta import archetype
from .meta.nxobjects import NxObject

__all__ = ['Operator']

# =============================================================================
# Operator base class
# =============================================================================


@archetype
class Operator(NxObject):
    pass
