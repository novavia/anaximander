#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines Anaximander base classes for use in applications.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from ._meta import BaseType, BaseObject

#==============================================================================
### Abstract base classes
#==============================================================================


class NxType(BaseType):
    """The parent metaclass to all NxObject."""
    pass


class NxObject(BaseObject, metaclass=NxType):
    """Parent class to all library and application objects."""
    pass
