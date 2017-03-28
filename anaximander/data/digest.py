#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines data digests.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc
from itertools import chain

import pandas as pd

from ..utilities import xprops, functions as fun
from ..meta import NxObject, prototype, archetype, typeattribute, \
    metacharacter
from .exceptions import DataError
from .fields import Str
from .schema import Schema, LinearSchema, TimeSchema
from .base import DataObject
from .series import NxSeries
from .frame import NxDataSequence
from .annotations import domain, Marker, Highlighter, Mark, Highlight

__all__ = []

# =============================================================================
# Schema classes
# =============================================================================


class DigestSchema(Schema):
    """Abstract blank schema for the Digest base class."""
    pass


class MarkSchema(DigestSchema):
    marker = Str(required=True)


class HighlightSchema(DigestSchema):
    _zero = None  # Placeholder for a 'zero' value in concrete classes
    prev_highlighter = Str(required=True)
    next_highlighter = Str(required=True)


class FloatMarkSchema(MarkSchema, LinearSchema):
    """Schema for FloatMarkDigest."""
    pass


class FloatHighlightSchema(HighlightSchema, LinearSchema):
    """Schema for FloatHighlightDigest."""
    _zero = 1e-9


class TimeMarkSchema(MarkSchema, TimeSchema):
    """Schema for TimeMarkDigest."""
    pass


class TimeHighlightSchema(HighlightSchema, TimeSchema):
    """Schema for TimeHighlightDigest."""
    _zero = pd.Timedelta(microseconds=1)


_domain_to_mark_schema = {None: MarkSchema,
                          'float': FloatMarkSchema,
                          'time': TimeMarkSchema}

_domain_to_high_schema = {None: HighlightSchema,
                          'float': FloatHighlightSchema,
                          'time': TimeHighlightSchema}

# =============================================================================
# Abstract base classes
# =============================================================================


class DigestError(DataError):
    """Specialize exception for Digests."""
    pass


@prototype
class Digest(NxDataSequence, schema=DigestSchema):
    """Abstract base class for Digests."""
    schema = metacharacter(validate=lambda s: issubclass(s, DigestSchema))

    @xprops.typedweakproperty(DataObject)
    def dataobject(self):
        """A weak reference to the target data object."""
        return None

    @property
    def context(self):
        return self.dataobject

    @context.setter
    def context(self, obj):
        self.dataobject = obj

    @context.deleter
    def context(self):
        del self.dataobject

    def __new__(cls, dataobject):
        return super().__new__(cls)

    def __init__(self, dataobject, data=None):
        """Requires a DataObject and a Marker instance."""
        super().__init__(data)
        self.dataobject = dataobject


class MarkDigest(Digest, schema=MarkSchema):
    """Base class and interface for Mark Digests."""

    def __new__(cls, dataobject, marker_type, data=None):
        if dataobject.empty:
            return EmptyMarkDigest(dataobject, marker_type, None)
        else:
            dom = domain(dataobject.index)
        try:
            schema = _domain_to_mark_schema[dom]
        except KeyError:
            msg = "Unrecognized survey domain for {0}"
            raise DigestError(msg.format(dataobject))           
        concrete_type = Digest[schema]
        return concrete_type(dataobject, marker_type, data)
    
    def __init__(self, dataobject, marker_type, data=None):
        super().__init__(dataobject, data)
        self.marker_type = marker_type

    @classmethod
    def from_marks(cls, dataobject, marker_type, marks):
        if not marks:
            return EmptyMarkDigest(dataobject, marker_type)
        marksdata = ((m.location, m.marker.name) for m in marks)
        schema = _domain_to_mark_schema[marks[0].domain]
        data = pd.DataFrame(marksdata, columns=schema.fieldnames)
        return cls(dataobject, marker_type, data)


class HighlightDigest(Digest, schema=HighlightSchema):
    """Base class and interface for Highlight Digests."""

    def __new__(cls, dataobject, highlighter_type, data=None):
        if dataobject.empty:
            return EmptyHighlightDigest(dataobject, highlighter_type, None)
        else:
            dom = domain(dataobject.index)
        try:
            schema = _domain_to_high_schema[dom]
        except KeyError:
            msg = "Unrecognized survey domain for {0}"
            raise DigestError(msg.format(dataobject))           
        concrete_type = Digest[schema]
        return concrete_type(dataobject, highlighter_type, data)

    def __init__(self, dataobject, highlighter_type, data=None):
        super().__init__(dataobject, data)
        self.highlighter_type = highlighter_type

    @classmethod
    def from_highlights(cls, dataobject, highlighter_type, highlights):
        if not highlights:
            return EmptyHighlightDigest(dataobject, highlighter_type)
        highsdata = list(chain(*[h.transitions() for h in highlights]))
        schema = _domain_to_high_schema[highlights[0].domain]
        data = pd.DataFrame(highsdata, columns=schema.fieldnames)
        diffs = data.iloc[:, 0].diff()
        if (diffs < -schema._zero).any():
            msg = "Cannot instantiate HighlightDigest from overlapping \
                   highlights."
            raise DigestError(msg)
        keepers = diffs > schema._zero
        keepers.iloc[0] = True
        data = data[keepers]
        next_highlights = data.prev_highlighter.shift(-1)
        next_highlights.iloc[-1] = 'blank'
        data['next_highlighter'] = next_highlights
        data = data[(data.prev_highlighter != 'blank') | \
                    (data.next_highlighter != 'blank')]
        return cls(dataobject, highlighter_type, data)

# =============================================================================
# Concrete Digest classes
# =============================================================================


class FloatMarkDigest(MarkDigest, schema=FloatMarkSchema):

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)


class FloatHighlightDigest(HighlightDigest, schema=FloatHighlightSchema):

    
    def __new__(cls, dataobject, highlighter_type, data=None):
        return NxObject.__new__(cls)


class TimeMarkDigest(MarkDigest, schema=TimeMarkSchema):

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)


class TimeHighlightDigest(HighlightDigest, schema=TimeHighlightSchema):
    
    def __new__(cls, dataobject, highlighter_type, data=None):
        return NxObject.__new__(cls)


class EmptyMarkDigest(MarkDigest, schema=MarkSchema, overwrite=True):

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)


class EmptyHighlightDigest(HighlightDigest, schema=HighlightSchema,
                           overwrite=True):

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)

# =============================================================================
# Survey classes
# =============================================================================


class SurveyError(DataError):
    """Specialized exception type for Surveys."""
    pass


class Survey(NxObject):
    """Wraps a function to produce a Digest."""

    def __init__(self, dataobject, *columns, **params):
        """Instantiates a survey object.

        Params:
            dataobject: an indexed dataobject, ie. NxSeries or NxDataFrame.
            *columns: optionally, the columns of a dataframe that will be
                surveyed. If dataobject is a series, this will be
                systematically ignored. Otherwise, admissible values can
                be integers, in which case the columns are selected by
                position, or strings (preferred), in wich case they are
                selected by names.
            **params: parameters that are passed to the survey method.
        """
        self.dataobject = dataobject
        self.columns = columns
        self.params = params

    @property
    def series(self):
        """An iterable of pandas series to pass to the the survey function."""
        if isinstance(self.dataobject, NxSeries):
            return self.dataobject.data
        else:
            df = self.dataobject.data
        if not self.columns:
            # Returns the first column of dataobject, assumed to be dataframe
            col = self.dataobject.schema.fieldnames[0]
            return df[col]
        elif all(isinstance(c, int) for c in self.columns):
            return (df.iloc[:, c] for c in self.columns)
        elif all(isinstance(c, str) for c in self.columns):
            return (df.loc[:, c] for c in self.columns)
        else:
            msg = "Ambiguous column definition in {0}".format(self)
            raise SurveyError(msg)

    @xprops.cachedproperty
    def digest(self):
        return self()

    @abc.abstractmethod
    def __call__(self, **kwargs):
        """Calls the survey function with optional runtime arguments."""
        pass


@archetype
class MarkSurvey(Survey):
    markertype = typeattribute(validate=fun.subcheck(Marker))

    def __call__(self, plot=False):
        """Calls the survey function with optional runtime arguments."""
        marks = self.__marks__(*self.series, **self.params)
        digest = MarkDigest.from_marks(self.dataobject, self.markertype, marks)
        self._digest = digest
        if plot:
            # TODO: insert plot routine
            pass
        return digest

    def mark(self, shade, loc):
        marker = self.markertype(shade)
        return Mark(self.dataobject, marker, loc)

    @abc.abstractmethod
    def __marks__(self, *series, **params):
        return []


@archetype
class HighlightSurvey(Survey):
    highlightertype = typeattribute(validate=fun.subcheck(Highlighter))

    def __call__(self, plot=False):
        """Calls the survey function with optional runtime arguments."""
        highlights = self.__highlights__(*self.series, **self.params)
        digest = HighlightDigest.from_highlights(self.dataobject,
                                                 self.highlightertype,
                                                 highlights)
        self._digest = digest
        if plot:
            # TODO: insert plot routine
            pass
        return digest

    def highlight(self, shade, lower, upper):
        highligther = self.highlightertype(shade)
        return Highlight(self.dataobject, highligther, lower, upper)

    @abc.abstractmethod
    def __highlights__(self, *series, **params):
        return []
