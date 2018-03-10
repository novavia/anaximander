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

from .meta import archetype, TypeParameter, typeproperty, typeinitmethod
from .structures import Structure
from .events import HardEvent, SoftEvent

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


@archetype
class StateTransition(HardEvent):
    """Event marking the transition of context to the state."""
    statetype = TypeParameter(type_=State.__metatype__)

    @typeinitmethod
    def validate_label(cls):
        if cls.label is None:
            return
        if cls.label not in cls.statetype.registry:
            msg = "Cannot create transition to undefined state label."
            raise ValueError(msg)


@archetype
class StatePhase(SoftEvent):
    """Event marking the transition of context to the state."""
    statetype = TypeParameter(type_=State.__metatype__)

    @typeinitmethod
    def validate_label(cls):
        if cls.label is None:
            return
        if cls.label not in cls.statetype.registry:
            msg = "Cannot create transition to undefined state label."
            raise ValueError(msg)


def statetype(cls):
    """Wraps archetype with additional functionalities for states."""
    archetype_ = archetype(cls)
    # Creates a default 'null' state
    archetype_['null']

    @archetype
    class Transition(StateTransition, statetype=archetype_):

        @property
        def state(self):
            return self.statetype[self.label]()

    @archetype
    class Phase(StatePhase, statetype=archetype_):
        pass

        @property
        def state(self):
            return self.statetype[self.label]()

    archetype_.Transition = Transition
    archetype_.Phase = Phase
    return archetype_
