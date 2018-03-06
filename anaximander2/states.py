#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the state archetype to define states.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from .meta import archetype, TypeParameter, typeproperty
from .structures import Structure

__all__ = ['State']

# =============================================================================
# Structure base class
# =============================================================================


@archetype
class State(Structure):
    label: str = TypeParameter(key=True)

    @typeproperty
    def substates(cls):
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
        return [c.label for c in cls.substates]

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))
