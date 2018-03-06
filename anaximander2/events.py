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

from .meta import archetype, TypeParameter, typeproperty
from .structures import Structure

__all__ = ['Event']

# =============================================================================
# Structure base class
# =============================================================================


@archetype
class Event(Structure):
    label: str = TypeParameter(key=True)

    def __init__(self, datetime):
        self.datetime = datetime

    @typeproperty
    def subtypes(cls):
        if cls.archetype is cls:
            base = cls.archetype.__basetype__
            states = []
            archetype = cls
        else:
            base = cls
            states = [cls]
            archetype = cls.archetype
        states += [c for c in archetype.clade if issubclass(c, base)]
        return states        

    @typeproperty
    def sublabels(cls):
        return [c.label for c in cls.subtypes]
