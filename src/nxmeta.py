#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module description.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

from abc import ABCMeta

import xprops

#==============================================================================
### NxMeta base class
#==============================================================================

class NxMeta(ABCMeta):
    """Parent class to all metaclasses in Anaximander."""


class NxBase(metaclass=NxMeta):
    """Base class for all Anaximander objects and types."""

    @xprops.cachedproperty
    def _nxregistries(self):
        return set()


class NxType(NxMeta, NxBase):
    """Parent class to all Anaximander types."""
