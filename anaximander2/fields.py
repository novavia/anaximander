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


__all__ = []

# =============================================================================
# Field base types
# =============================================================================


class NxField:
    pass


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

    def __init__(self, categories):
        self.categories = tuple(categories)


class State(Categorical):
    """State field type, requiring a state type.

    The admissible categories are the state type and its subtypes,
    passed by string references.
    """

    def __init__(self, stype):
        self.stype = stype


class EventType(Categorical):
    """Field specifying event type, requiring an event type.

    The admissible categories are the event type and its subtypes,
    passed by string references.
    """

    def __init__(self, etype):
        self.etype = etype


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

    def __init__(self, otype):
        self.otype = otype
