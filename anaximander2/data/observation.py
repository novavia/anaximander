#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the data observation archetype for scalar values.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from ..utilities import functions as fun
from ..fields import NxField
from ..meta import archetype, TypeParameter
from .base import DataObject


__all__ = []

# =============================================================================
# Base types
# =============================================================================


@archetype
class Observation(DataObject):
    """Base class for all data objects."""
    ftype = TypeParameter(covariant_from=NxField)

    
