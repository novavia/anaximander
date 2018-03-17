#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the event archetype to define event types and events.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from abc import abstractproperty
from datetime import timedelta

from .meta import archetype, TypeParameter, typeproperty
from .observations import NxObservation

__all__ = ['NxSample']

# =============================================================================
# Sample base class
# =============================================================================


# TODO: turn NxSample into an archetype with a physical quantity type parameter
class NxSample(NxObservation):
    """Abstract base class for events."""

    def __init__(self, data, datetime, object=None, unit=None):
        super().__init__(object)
        self._data = data
        self.datetime = datetime
        self.unit = unit

    @property
    def locus(self):
        return self.datetime

    @property
    def data(self):
        return self.data

    @property
    def metadata(self):
        return {'unit': self.unit}
