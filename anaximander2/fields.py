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
    """NxFields are schema descriptors.

    The objects don't feature much functionality as they primarily serve
    as interface specification between objects, data and storage.
    NxField inherit name, cls and registration_id attributes from Registrable.
    NxField can be marked as index: index fields are featured in a schema's
    index whereas other fields are columns (or column families in the case
    of a multi-schema).
    Additionally, the sequencer property specifies that a field
    defines an order between records. This property is used for database
    indexing, and there can be at most one sequencer index field per schema.
    """
    __registry__ = '__nxfields__'
    __counter__ = count()

    def __init__(self, index=False, sequencer=False, name=None, cls=None,
                 registration_id=None):
        super().__init__(name, cls, registration_id)
        self.sequencer = sequencer
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

    def __init__(self, categories, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        super().__init__(index, sequencer, name, cls, registration_id)
        self.categories = tuple(categories)


class State(Categorical):
    """State field type, requiring a state archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, index, sequencer, name, cls,
                         registration_id)


class EventType(Categorical):
    """Field specifying event type, requiring an event archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, index, sequencer, name, cls,
                         registration_id)


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

    def __init__(self, otype, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        super().__init__(index, sequencer, name, cls, registration_id)
        self.otype = otype
