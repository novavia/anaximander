#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines base classes for Anaximander's base layer, which is
a collection of libraries that implement Anaximander's meta layer but are
somewhat optional to writing applications.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import nxmeta

#==============================================================================
### Abstract base classes
#==============================================================================


class NxBaseType(nxmeta.NxType):
    """The parent metaclass to all NxBaseObject."""
    pass


class NxBaseObject(nxmeta.NxObject, metaclass=NxBaseType):
    """Parent class to all objects in the base layer and application layers."""
    pass
