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

from .meta import archetype, TypeParameter, TypeAttribute, typeproperty, \
    typeinitmethod, call, typenamemethod
from .structures import NxStructure
from .observations import NxObservation
from .events import HardEvent, SoftEvent

__all__ = ['NxState']

# =============================================================================
# State base class
# =============================================================================


class NxStateError(Exception):
    """Customized exception class for NxState-related errors."""
    pass


@archetype
class NxState(NxStructure):
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

    @classmethod
    def transition(cls, datetime, object=None, label=None):
        if label is not None:
            if label not in cls.sublabels:
                msg = f"Improper label {label} passed to {cls}.transition."
                raise NxStateError(msg)
            else:
                statetype = cls.archetype[label]
        else:
            statetype = cls
        Transition = cls.archetype.Transition[statetype]
        return Transition(datetime, object=object)

    @classmethod
    def phase(cls, start, stop, object=None, label=None):
        if label is not None:
            if label not in cls.sublabels:
                msg = f"Improper label {label} passed to {cls}.phase."
                raise NxStateError(msg)
            else:
                statetype = cls.archetype[label]
        else:
            statetype = cls
        Phase = cls.archetype.Phase[statetype]
        return Phase(start, stop, object=object)

    @classmethod
    def status(cls, datetime, object=None, label=None):
        if label is not None:
            if label not in cls.sublabels:
                msg = f"Improper label {label} passed to {cls}.transition."
                raise NxStateError(msg)
            else:
                statetype = cls.archetype[label]
        else:
            statetype = cls
        Status = cls.archetype.Status[statetype]
        return Status(datetime, object=object)

    @typenamemethod
    def name_type(mcl, basetype, *traits, **kwargs):
        try:
            return kwargs['label'].title() + mcl.__basename__
        except (KeyError, AttributeError):
            return mcl.__basename__


def statelabel(cls):
    """Label attribution function for state-based observation types."""
    try:
        return cls.statetype.label
    except AttributeError:
        return None


class StateObservation:
    """Mix-in class for Transition, Phase and Status."""

    @property
    def state(self):
        return self.statetype()


class StateTransition(HardEvent, StateObservation):
    """Event marking the transition of context to the state."""

    @typeinitmethod
    def set_label(cls):
        cls._label = cls.statetype.label

    @typenamemethod
    def name_type(mcl, basetype, *traits, **kwargs):
        try:
            return kwargs['statetype'].label.title() + mcl.__basename__
        except (KeyError, AttributeError):
            return mcl.__basename__


class StatePhase(SoftEvent, StateObservation):
    """Event marking the transition of context to the state."""

    @typeinitmethod
    def set_label(cls):
        cls._label = cls.statetype.label

    @typenamemethod
    def name_type(mcl, basetype, *traits, **kwargs):
        try:
            return kwargs['statetype'].label.title() + mcl.__basename__
        except (KeyError, AttributeError):
            return mcl.__basename__


class BaseStatus(NxObservation, StateObservation):

    def __init__(self, datetime, object=None, **params):
        super().__init__(object, **params)
        self.datetime = datetime

    @property
    def locus(self):
        return self.datetime

    @typeinitmethod
    def set_label(cls):
        cls._label = cls.statetype.label

    @typenamemethod
    def name_type(mcl, basetype, *traits, **kwargs):
        try:
            return kwargs['statetype'].label.title() + mcl.__basename__
        except (KeyError, AttributeError):
            return mcl.__basename__

NxState.Transition = StateTransition
NxState.Phase = StatePhase
NxState.Status = BaseStatus


def statetype(cls):
    """Wraps archetype with additional functionalities for states."""
    archetype_ = archetype(cls)

    # Creates a default 'null' state
    class Null(archetype_, label='null'):
        pass

    @archetype
    class Transition(StateTransition, statetype=archetype_):
        statetype = TypeParameter(covariant_from=archetype_)
        label: str = TypeAttribute()

    @archetype
    class Phase(StatePhase, statetype=archetype_):
        statetype = TypeParameter(covariant_from=archetype_)
        label: str = TypeAttribute()

    @archetype
    class Status(BaseStatus, statetype=archetype_):
        statetype = TypeParameter(covariant_from=archetype_)
        label: str = TypeAttribute()

    archetype_.Transition = Transition
    archetype_.Phase = Phase
    archetype_.Status = Status
    return archetype_
