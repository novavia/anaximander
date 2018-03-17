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

from collections import Mapping
from itertools import count

from ..meta.nxdescriptors import Registrable


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

    def match_type(self, typemap):
        """Returns the type corresponding to self in a type map."""
        for t in type(self).__mro__:
            try:
                match = typemap[t]
            except KeyError:
                continue
            else:
                if isinstance(match, TypeMapper):
                    return match(self)
                else:
                    return match
        raise TypeError(f"Could not match {self} in {typemap}.")


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
    """Categorical field type, with string-defined categories.

    The ordered flag specifies whether the categories define a natural sort
    order.
    """

    def __init__(self, categories, ordered=False, index=False, sequencer=False,
                 name=None, cls=None, registration_id=None):
        super().__init__(index, sequencer, name, cls, registration_id)
        self.categories = tuple(categories)
        self.ordered = ordered


class State(Categorical):
    """State field type, requiring a state archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, False, index, sequencer, name,
                         cls, registration_id)


class EventType(Categorical):
    """Field specifying event type, requiring an event archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, False, index, sequencer, name,
                         cls, registration_id)


class DateTimeBase(NxField):
    """Base class for date & time fields."""

    def __init__(self, tz=None, index=False, sequencer=False, name=None,
                 cls=None, registration_id=None):
        self.tz = tz
        super().__init__(index, sequencer, name, cls, registration_id)


class Date(DateTimeBase):
    """Date field type."""
    pass


class Time(DateTimeBase):
    """Time field type."""
    pass


class DateTime(DateTimeBase):
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

# =============================================================================
# Type map class
# =============================================================================


class TypeMapper:
    """A callable designed to map a field to a parametric type.

    The supplied function should take a single field argument.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, field):
        return self.func(field)


class TypeMap(Mapping):
    """A mapping of field types to another type system."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return self._mapping.__iter__()

    def __len__(self):
        return self._mapping.__len__()

    def __call__(self, field):
        """Matches a given field to a mapped type."""
        return field.match_type(self)
