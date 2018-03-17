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

from .meta import archetype, TypeParameter, typeproperty, typenamemethod
from .observations import NxObservation

__all__ = ['HardEvent', 'SoftEvent']

# =============================================================================
# Event base classes
# =============================================================================


class EventBase(NxObservation):
    """Abstract base class for events."""
    label: str = TypeParameter(key=True)

    @abstractproperty
    def duration(self):
        return None

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

    @typenamemethod
    def name_type(mcl, basetype, *traits, **kwargs):
        try:
            return kwargs['label'].title() + mcl.__basename__
        except (KeyError, AttributeError):
            return mcl.__basename__


@archetype
class HardEvent(EventBase):
    """A duration-less event with a unique timestamp."""

    def __init__(self, datetime, object=None):
        super().__init__(object)
        self.datetime = datetime

    @property
    def locus(self):
        return self.datetime

    @property
    def duration(self):
        return timedelta(0)


@archetype
class SoftEvent(EventBase):
    """An event with a start and stop time.

    Following Python's general indexing conventions, the time interval
    is closed on the start side and open on the stop side.
    """

    def __init__(self, start, stop, object=None):
        super().__init__(object)
        self.start = start
        self.stop = stop

    @property
    def locus(self):
        return (self.start, self.stop)

    @property
    def duration(self):
        return self.stop - self.start
