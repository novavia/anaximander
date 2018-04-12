#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines basic column types used in data structures.

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
# Column base types
# =============================================================================


class NxColumn(Registrable):
    """NxColumns are schema descriptors.

    The objects don't feature much functionality as they primarily serve
    as interface specification between objects, data and storage.
    NxColumn inherit name, cls and registration_id attributes from Registrable.
    NxColumn can be marked as index: index columns are featured in a schema's
    index whereas other are simple columns (or column families in the case
    of a multi-schema). Index columns come in two flavors: sequential or
    nominal. A sequential index sets ordering between records that otherwise
    share the same nominal index values. There can be at most one sequential
    index column in a schema.
    """
    __registry__ = '__nxcolumns__'
    __counter__ = count()

    def __init__(self, index=None, name=None, cls=None,
                 registration_id=None):
        super().__init__(name, cls, registration_id)
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


class Numeric(NxColumn):
    """Base class for numeric types."""
    pass


class Integer(Numeric):
    """Integer column type."""
    pass


class Float(Numeric):
    """Float column type."""
    pass


class String(NxColumn):
    """Base class for text-based field columns."""
    pass


class Text(String):
    """Text column type."""
    pass


class Categorical(String):
    """Categorical column type, with string-defined categories.

    The ordered flag specifies whether the categories define a natural sort
    order.
    """

    def __init__(self, categories, ordered=False, index=None,
                 name=None, cls=None, registration_id=None):
        super().__init__(index, name, cls, registration_id)
        self.categories = tuple(categories)
        self.ordered = ordered


class State(Categorical):
    """State column type, requiring a state archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=None, name=None,
                 cls=None, registration_id=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, False, index, name,
                         cls, registration_id)


class EventType(Categorical):
    """Column specifying event type, requiring an event archetype.

    The admissible categories are the archetype's labels,
    passed by string references.
    """

    def __init__(self, archetype, index=False, name=None,
                 cls=None, registration_id=None):
        self.state_type = archetype
        super().__init__(archetype.sublabels, False, index, name,
                         cls, registration_id)


class DateTimeBase(NxColumn):
    """Base class for date & time columns."""

    def __init__(self, tz=None, index=False, name=None,
                 cls=None, registration_id=None):
        self.tz = tz
        super().__init__(index, name, cls, registration_id)


class Date(DateTimeBase):
    """Date column type."""
    pass


class Time(DateTimeBase):
    """Time column type."""
    pass


class DateTime(DateTimeBase):
    """DateTime column type."""
    pass


class Timestamp(Integer):
    """Timestamp column type."""
    pass


class ObjectID(Integer):
    """Column type for integer references to objects.

    Requires an object type at instantiation.
    """

    def __init__(self, otype, index=None, name=None,
                 cls=None, registration_id=None):
        super().__init__(index, name, cls, registration_id)
        self.otype = otype

# =============================================================================
# Type map class
# =============================================================================


class TypeMapper:
    """A callable designed to map a column to a parametric type.

    The supplied function should take a single column argument.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, column):
        return self.func(column)


class TypeMap(Mapping):
    """A mapping of column types to another type system."""

    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def __getitem__(self, key):
        return self._mapping[key]

    def __iter__(self):
        return self._mapping.__iter__()

    def __len__(self):
        return self._mapping.__len__()

    def __call__(self, column):
        """Matches a given column to a mapped type."""
        return column.match_type(self)
