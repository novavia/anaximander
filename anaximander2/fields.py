#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines basic field types used in data structures.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from itertools import count

from .meta.nxdescriptors import Registrable


__all__ = []

# =============================================================================
# Field base types
# =============================================================================


class NxField(Registrable):
    __registry__ = '__nxfields__'
    __counter__ = count()

    def __init__(self, index=False, name=None, cls=None):
        super().__init__(name, cls)
        self.index = index


class Numeric(NxField):
    """Base class for numeric types."""
    pass


class Integer(Numeric):
    """Integer field type."""
    pass


class Float(Numeric):
    """Float field type."""
    pass


class String(NxField):
    """Base class for text-based fields."""
    pass


class Text(String):
    """Text field type."""
    pass


class Categorical(String):
    """Categorical field type, with string-defined categories."""

    def __init__(self, categories, index=False, name=None, cls=None):
        super().__init__(index, name, cls)
        self.categories = tuple(categories)


class State(Categorical):
    """State field type, requiring a state archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, name=None, cls=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, index, name, cls)


class EventType(Categorical):
    """Field specifying event type, requiring an event archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, name=None, cls=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, index, name, cls)


class Date(NxField):
    """Date field type."""
    pass


class Time(NxField):
    """Time field type."""
    pass


class DateTime(NxField):
    """DateTime field type."""
    pass


class Timestamp(Integer):
    """Timestamp field type."""
    pass


class ObjectID(Integer):
    """Field type for integer references to objects.

    Requires an object type at instantiation.
    """

    def __init__(self, otype, index=False, name=None, cls=None):
        super().__init__(index, name, cls)
        self.otype = otype
