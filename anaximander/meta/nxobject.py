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

from .nxtype import NxType
from ..registries.folios import RegistrableObject

# =============================================================================
# NxType declaration
# =============================================================================


class NxObject(RegistrableObject, metaclass=NxType):
    """The Anaximander base object class."""
    pass
