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
from .structures import Structure

__all__ = ['HardEvent', 'SoftEvent']

# =============================================================================
# Structure base class
# =============================================================================


class EventBase(Structure):
    """Abstract base class for events.

    Params:
        context (object): the context, typically a data model entity, to which
            the event applies.
    """
    label: str = TypeParameter(key=True)

    def __init__(self, context=None):
        self.context = context

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


@archetype
class HardEvent(EventBase):
    """A duration-less event with a unique timestamp."""

    def __init__(self, datetime, context=None):
        super().__init__(context)
        self.datetime = datetime

    @property
    def duration(self):
        return timedelta(0)


@archetype
class SoftEvent(EventBase):
    """An event with a start and stop time.

    Following Python's general indexing conventions, the time interval
    is closed on the start side and open on the stop side.
    """

    def __init__(self, start, stop, context=None):
        super().__init__(context)
        self.start = start
        self.stop = stop

    @property
    def duration(self):
        return self.stop - self.start
