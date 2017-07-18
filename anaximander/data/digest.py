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
from collections import OrderedDict
from collections.abc import Mapping
from itertools import chain

import numpy as np
import pandas as pd

from ..utilities import xprops, functions as fun
from ..meta import NxObject, prototype, archetype, typeattribute, \
    metacharacter
from .exceptions import DataError
from .fields import Str
from .schema import Schema, LinearSchema, TimeSchema, PostChartSchema, \
    SectionChartSchema, EventLogSchema, PhaseLogSchema, ClipLogSchema
from .base import DataObject
from .record import NxRecord
from .series import NxSeries
from .frame import NxDataSequence
from .annotations import Marker, Highlighter, Mark, Highlight
from .plot import plot_marks, plot_highlights

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


MarkRecord = NxRecord[MarkSchema]
HighlightRecord = NxRecord[HighlightSchema]
FloatMarkRecord = NxRecord[FloatMarkSchema]
FloatHighlightRecord = NxRecord[FloatHighlightSchema]
TimeMarkRecord = NxRecord[TimeMarkSchema]
TimeHighlightRecord = NxRecord[TimeHighlightSchema]


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

    def to_frame(self):
        pass


class MarkDigest(Digest, schema=MarkSchema):
    """Base class and interface for Mark Digests."""

    def __new__(cls, dataobject, marker_type, data=None):
        domain = dataobject.domain
        try:
            schema = _domain_to_mark_schema[domain]
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
        marks = list(marks)
        if not marks:
            return EmptyMarkDigest(dataobject, marker_type)
        marksdata = ((m.location, m.marker.shade) for m in marks)
        schema = _domain_to_mark_schema[marks[0].domain]
        data = pd.DataFrame(marksdata, columns=schema.fieldnames)
        return cls(dataobject, marker_type, data)

    def marks(self, *shades):
        """Returns the list of marks, with optional shade filter.

        By default, blank shades are ignored.
        """
        dataobject = self.dataobject
        marker_type = self.marker_type

        def mark(record):
            loc, shade = record.as_tuple()
            return Mark(dataobject, marker_type(shade), loc)
        marks = [mark(r) for r in self]

        def filter_(mark):
            if not shades:
                return mark.shade != 'blank'
            else:
                return mark.shade in shades
        return list(filter(filter_, marks))

    def records(self, *shades):
        """Returns the records from dataobject corresponding to marks."""
        return [self.dataobject.loc[m.location] for m in self.marks(*shades)]

    def sub(self, data):
        """Returns an object with identical attributes except for mark data."""
        return type(self)(self.dataobject, self.marker_type, data)

    def shades(self, *shades):
        """Returns a set of shades found in self, with optional fiter.

        By default the blank shade is ommitted.
        """
        datashades = set(self.data.marker.unique())
        if not shades:
            return datashades - {'blank'}
        else:
            return datashades & set(shades)

    def shadegroups(self, *shades):
        """Returns a list of digests with single markers.

        Optionally, a set of shades can be specified to downselect.
        By default, the blank shade is ommitted.
        """
        grouped = self.data.groupby('marker')
        keys = set(grouped.groups.keys())
        if not shades:
            keys -= {'blank'}
        else:
            keys &= set(shades)
        groups = (grouped.get_group(k) for k in keys)
        return [self.sub(g) for g in groups]

    def plot(self, y, ax='new', **kwargs):
        return plot_marks(self, y, ax, **kwargs)


class HighlightDigest(Digest, schema=HighlightSchema):
    """Base class and interface for Highlight Digests."""

    def __new__(cls, dataobject, highlighter_type, data=None,
                normalize=True):
        domain = dataobject.domain
        try:
            schema = _domain_to_high_schema[domain]
        except KeyError:
            msg = "Unrecognized survey domain for {0}"
            raise DigestError(msg.format(dataobject))
        concrete_type = Digest[schema]
        return concrete_type(dataobject, highlighter_type, data)

    def __init__(self, dataobject, highlighter_type, data=None,
                 normalize=True):
        super().__init__(dataobject, data)
        if normalize:
            self._data = self.normalize(self._data)
        self.highlighter_type = highlighter_type

    @classmethod
    def normalize(cls, data):
        """Normalizes a dataframe that already contain transitions."""
        if data.empty:
            return data
        # Check that there are no overlaps between highlights, raise otherwise
        diffs = data.iloc[:, 0].diff()
        if (diffs < -cls.schema._zero).any():
            msg = "Cannot instantiate HighlightDigest from overlapping \
                   highlights."
            raise DigestError(msg)
        # Eliminates useless transitions where prev == next
        return data[data.prev_highlighter != data.next_highlighter]

    @classmethod
    def from_highlights(cls, dataobject, highlighter_type, highlights):
        domain = dataobject.domain
        try:
            schema = _domain_to_high_schema[domain]
        except KeyError:
            msg = "Unrecognized survey domain for {0}"
            raise DigestError(msg.format(dataobject))
        # Eliminates zero-length highlights that provide no information
        highlights = [h for h in highlights if h.length > schema._zero]
        highlights.sort(key=lambda h: h.lower)
        if not highlights:
            return EmptyHighlightDigest(dataobject, highlighter_type)
        # First, turns highlights into transitions
        highsdata = list(chain(*[h.transitions() for h in highlights]))
        # Make an underlying pandas dataframe from the transitions
        schema = _domain_to_high_schema[highlights[0].domain]
        data = pd.DataFrame(highsdata, columns=schema.fieldnames)
        # Eliminates meaningless transitions due to consecutive highlights
        # that share a common boundary
        diffs = data.iloc[:, 0].diff()
        keepers = diffs > schema._zero
        keepers.iloc[0] = True
        data = pd.DataFrame(data[keepers])
        # The elimination of rows require resetting the transitions
        # We shift the prev_highlighter to populate next_highlighter
        next_highlights = data.prev_highlighter.shift(-1)
        next_highlights.iloc[-1] = 'blank'
        data['next_highlighter'] = next_highlights
        return cls(dataobject, highlighter_type, data)

    @classmethod
    def singleshade(cls, dataobject, highlighter_type, shade, start, end):
        """Returns a digest made of a single shade from start to end.

        Params:
            dataobject: the surveyed dataobject
            highlighter_type: a highlighter type.
            shade: admissible shade for the supplied highlighter type.
            start: a start coordinate or timestamp.
            end: end coordinate or timestamp.
        """
        highlight = Highlight(dataobject, highlighter_type(shade), start, end)
        return cls.from_highlights(dataobject, highlighter_type, [highlight])

    def highlights(self, *shades):
        """Returns the list of highlights, with optional filter.

        By default, blank shades are ignored.
        """
        dataobject = self.dataobject
        highlighter_type = self.highlighter_type

        def highlight(pair):
            """Returns a highlight from a successive record pair."""
            left, right = pair
            lower, _, shade = left.as_tuple()
            upper, *_ = right.as_tuple()
            return Highlight(dataobject, highlighter_type(shade), lower, upper)
        highlights = [highlight(pair) for pair in fun.pairwise(self)]

        def filter_(highlight):
            if not shades:
                return highlight.shade != 'blank'
            else:
                return highlight.shade in shades
        return list(filter(filter_, highlights))

    def sessions(self, *shades):
        """Returns the list of sessions corresponding to highlights."""
        highlights = self.highlights(*shades)
        return [self.dataobject.loc[h.lower:h.upper] for h in highlights]

    def sub(self, data):
        """Returns an object with identical attributes except for data."""
        return type(self)(self.dataobject, self.highlighter_type, data)

    def shades(self, *shades):
        """Returns a set of shades found in self, with optional fiter.

        By default the blank shade is ommitted.
        """
        prev_shades = set(self.data.prev_highlighter.unique())
        next_shades = set(self.data.next_highlighter.unique())
        datashades = prev_shades | next_shades
        if not shades:
            return datashades - {'blank'}
        else:
            return datashades & set(shades)

    def shadegroups(self, *shades):
        """Returns a list of digests with single highlighter.

        Optionally, a set of shades can be specified to downselect.
        By default, the blank shade is ommitted.
        The groups are meant to be read in pairs, where each pair
        corresponds to a highlight whose shade is the next / prev
        highlighter in the first / second element of the pair.
        """
        grouped_prev = self.data.groupby('prev_highlighter')
        grouped_next = self.data.groupby('next_highlighter')
        keys = grouped_next.groups.keys() | grouped_prev.groups.keys()
        if not shades:
            keys -= {'blank'}
        else:
            keys &= set(shades)
        groups = []
        for k in keys:
            try:
                gprev = grouped_prev.get_group(k)
                gnext = grouped_next.get_group(k)
            except KeyError:
                continue
            if gprev.index[0] < gnext.index[0]:
                gprev = gprev.iloc[1:]
            group = gprev.merge(gnext, 'outer')
            groups.append(group)
        return [self.sub(g) for g in groups]

    def plot(self, y0, y1, ax='new', **kwargs):
        ax = plot_highlights(self, y0, y1, ax, **kwargs)
        if not self.empty:
            pd.Series(y0, index=self.index).plot(ax=ax, alpha=0)
        return ax

# =============================================================================
# Concrete Digest classes
# =============================================================================


class FloatMarkDigest(MarkDigest, schema=FloatMarkSchema):
    domain = 'float'

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)

    def to_frame(self, schema=PostChartSchema):
        cls = NxDataSequence[schema]
        data = self.data
        data['post_type'] = data['marker']
        return cls(data, context=self.context)


class FloatHighlightDigest(HighlightDigest, schema=FloatHighlightSchema):
    domain = 'float'

    def __new__(cls, dataobject, highlighter_type, data=None,
                normalize=True):
        return NxObject.__new__(cls)

    def to_frame(self, schema=SectionChartSchema):
        cls = NxDataSequence[schema]
        data = self.data
        data['prev_type'] = data['prev_highlighter']
        data['next_type'] = data['next_highlighter']
        return cls(data, context=self.context)


class TimeMarkDigest(MarkDigest, schema=TimeMarkSchema):
    domain = 'time'

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)

    def to_frame(self, schema=EventLogSchema):
        cls = NxDataSequence[schema]
        data = self.data
        data['event_type'] = data['marker']
        return cls(data, context=self.context)


class TimeHighlightDigest(HighlightDigest, schema=TimeHighlightSchema):
    domain = 'time'

    def __new__(cls, dataobject, highlighter_type, data=None,
                normalize=True):
        return NxObject.__new__(cls)

    def to_frame(self, schema=PhaseLogSchema):
        cls = NxDataSequence[schema]
        data = self.data
        data['prev_state'] = data['prev_highlighter']
        data['next_state'] = data['next_highlighter']
        return cls(data, context=self.context)


class PhaseTransitions(NxDataSequence):
    """Specialized NxDataSequence for phase transitions."""
    schema = PhaseLogSchema

    @classmethod
    def monophase(cls, tract, start, end, state, context=None):
        """Makes an empty log and sets the unique state."""
        result = tract.Sequence(context=context, timestamp=(start, end))
        result.unique = state
        return result

    @xprops.settablecachedproperty
    def highlighter_type(self):
        try:
            return self.schema.highlighter_type
        except AttributeError:
            return None

    @property
    def unique(self):
        """A unique state throughout the sequential range, or None."""
        if self.empty:
            return getattr(self, '_unique', None)
        values = self.data[['prev_state', 'next_state']].values
        states = np.unique(values)
        if len(states) == 1:
            return states[0]
        else:
            return None

    @unique.setter
    def unique(self, value):
        self._unique = value

    def state(self, timestamp):
        """Returns the state at timestamp."""
        if self.lower is not None:
            if timestamp < self.lower:
                msg = "Cannot determine state prior to the logs' lower bound."
                raise IndexError(msg)
        if self.upper is not None:
            if timestamp > self.upper:
                msg = "Cannot determine state after the log's upper bound."
        if self.empty:
            return self.unique
        prev_data = self.data.loc[:timestamp]
        next_data = self.data.loc[timestamp:]
        try:
            return prev_data.iloc[-1].next_state
        except IndexError:
            return next_data.iloc[0].prev_state

    def as_digest(self):
        if self.highlighter_type is None:
            msg = "Cannot convert / plot a PhaseTransitions frame without " + \
                "assigning a highlighter type."
            raise DigestError(msg)
        df = self.data
        df['prev_highlighter'] = df['prev_state']
        df['next_highlighter'] = df['next_state']
        df = df[['timestamp', 'prev_highlighter', 'next_highlighter']]
        df = df.replace('N/A', 'blank')
        if self.lower is not None:
            transition = OrderedDict([('timestamp', self.lower),
                                      ('prev_highlighter', 'blank'),
                                      ('next_highlighter',
                                       self.state(self.lower))])
            if df.empty:
                df = pd.DataFrame(transition, index=[self.lower])
            elif self.lower not in df.index:
                df.loc[self.lower] = list(transition.values())
        if self.upper is not None:
            transition = OrderedDict([('timestamp', self.upper),
                                      ('prev_highlighter',
                                       self.state(self.upper)),
                                      ('next_highlighter', 'blank')])
            if df.empty:
                df = pd.DataFrame(transition, index=[self.upper])
            elif self.upper not in df.index:
                df.loc[self.upper] = list(transition.values())
        df.sort_index(inplace=True)
        return TimeHighlightDigest(self, self.highlighter_type, df,
                                   normalize=False)

    def to_series(self):
        """Returns a timeseries of next states from lower to upper."""
        i0, s0 = self.lower, self.state(self.lower)
        i1, s1 = self.upper, self.state(self.upper)
        if self.empty:
            index = [i0, i1]
            states = [s0, s1]
        else:
            index, states = list(self.index), list(self.next_state())
            if self.lower < self.index[0]:
                i0, s0 = self.lower, self.iloc[0].prev_state
                index = [i0] + index
                states = [s0] + states
            if self.upper > self.index[-1]:
                i1, s1 = self.upper, 'N/A'
                index = index + [i1]
                states = states + [s1]
        return pd.Series(states, index=index)

    def crop(self, start=None, end=None):
        """Crops the transition logs.

        Identical to the generic method, except that a state metadata
        may be added if the dataframe ends up empty.
        """
        result = NxDataSequence.crop(self, start, end)
        if not result.empty:
            if result.index[0] == result.lower:
                result._data = result.data.iloc[1:]
        if not result.empty:
            if result.index[-1] == result.upper:
                result._data = result.data.iloc[:-1]
        if not self.empty and result.empty:
            prev_data = self.data.loc[:start]
            next_data = self.data.loc[end:]
            try:
                state = prev_data.iloc[-1].next_state
            except IndexError:
                state = next_data.iloc[0].prev_state
            result.unique = state
        elif self.empty:
            result.unique = self.unique
        return result

    def plot(self, series=None, *, y0=None, y1=None, ax='new', **kwargs):
        """Plots phase transitions as highlights, against optional series.

        Params:
            series: an optional series aginst which to plot the transitions.
            y0, y1: y-axis levels at which to plot the transitions. If series
            is not None, it will be used by default to establish y-axis
            levels, but y0, y1 will override those defaults. If series is None,
            y0 and y1 default to 0 and 1, respectively.
            ax: optional matplotlib ax object.
            **kwargs: optional arguments passed on to the plot method (still
            undefined).
        Returns:
            Matplotlib ax object.
        """
        if series is not None and len(series) > 1:
            mean = series.data.mean()
            std = series.data.std()
            y0 = fun.get(y0, mean - 0.5 * std)
            y1 = fun.get(y1, mean + 0.5 * std)
        else:
            y0 = fun.get(y0, 0)
            y1 = fun.get(y1, 1)
        ax = self.as_digest().plot(y0=y0, y1=y1, ax=ax, **kwargs)
        if series is not None:
            series.plot(ax=ax)
        return ax

PhaseTransitionRecord = NxRecord[PhaseLogSchema]


class ClipExcerpt(NxDataSequence):
    """Specialized NxDataSequence for regularly sampled clips."""
    schema = ClipLogSchema

    @property
    def freq(self):
        return self.schema().freq

    def fill(self, state, start=None, end=None):
        """Fills self with supplied state between start and end.

        If self's index extends beyond start and end, this method does
        nothing. Otherwise it inserts additional clips at regular interval
        (per the schema's sampling frequency) to fill the gaps.
        """
        if self.empty:
            msg = "Cannot fill an empty ClipExcerpt sequence."
            raise DataError(msg)
        df = self.data
        if start is not None:
            if df.index[0] > start:
                ix_range = pd.date_range(start, df.index[0], freq=self.freq,
                                         closed='left')
                first_ix = df.loc[start:].index[0]
                record = df.loc[first_ix].copy()
                record[self.schema.__clip__] = state
                for ix in ix_range:
                    r = record.copy()
                    r['timestamp'] = ix
                    df.loc[ix] = r
                df.sort_index(inplace=True)
                df.sort_index(inplace=True)
        if end is not None:
            if df.index[-1] < end:
                ix_range = pd.date_range(df.index[-1], end, freq=self.freq,
                                         closed='right')
                last_ix = df.loc[:end].index[-1]
                record = df.loc[last_ix].copy()
                record[self.schema.__clip__] = state
                for ix in ix_range:
                    r = record.copy()
                    r['timestamp'] = ix
                    df.loc[ix] = r
                df.sort_index(inplace=True)
        return type(self)(df, context=self.context)


class EmptyMarkDigest(MarkDigest, schema=MarkSchema, overwrite=True):

    def __new__(cls, dataobject, marker_type, data=None):
        return NxObject.__new__(cls)


class EmptyHighlightDigest(HighlightDigest, schema=HighlightSchema,
                           overwrite=True):

    def __new__(cls, dataobject, marker_type, data=None, normalize=True):
        return NxObject.__new__(cls)

# =============================================================================
# Survey classes
# =============================================================================


class SurveyError(DataError):
    """Specialized exception type for Surveys."""
    pass


class Survey(NxObject):
    """Wraps a function to produce a Digest."""
    # Indicates whether __highlights__ supports empty series.
    survey_empty = False

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

    @xprops.cachedproperty
    def series(self):
        """A list of NxSeries / pd.Series passed to the the survey function."""
        if self.dataobject.empty:
            if not self.columns:
                return [pd.Series()]
            else:
                return [pd.Series() for c in self.columns]
        if isinstance(self.dataobject, NxSeries):
            return [self.dataobject]
        if not self.columns:
            # Returns the first column of dataobject, assumed to be dataframe
            col = self.dataobject.schema.fieldnames[0]
            return [getattr(self.dataobject, col)()]
        elif all(isinstance(c, int) for c in self.columns):
            cols = [self.dataobject.schema.fieldnames[i] for i in self.columns]
        elif all(isinstance(c, str) for c in self.columns):
            cols = self.columns
        else:
            msg = "Ambiguous column definition in {0}".format(self)
            raise SurveyError(msg)
        return [getattr(self.dataobject, col)() for col in cols]

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
        """Calls the survey function with optional runtime arguments.

        The optional plot argument overlays the digest on the primary
        series before returning. The argument can take a dictionary whose
        key-value pairs will be passed to the plotting function.
        """
        marks = self.marks()
        digest = MarkDigest.from_marks(self.dataobject, self.markertype, marks)
        self._digest = digest
        if plot is not False:
            if isinstance(plot, Mapping):
                self._plot(**plot)
            else:
                self._plot()
        return digest

    def mark(self, shade, loc):
        marker = self.markertype(shade)
        return Mark(self.dataobject, marker, loc)

    def marks(self, *shades):
        """Returns the marks from survey, with optional filter."""
        series = [s.data if isinstance(s, NxSeries) else s
                  for s in self.series]
        if any(s.empty for s in series) and not self.survey_empty:
            marks = []
        else:
            marks = self.__marks__(*series, **self.params)

        def filter_(mark):
            if not shades:
                return True
            else:
                return mark.shade in shades
        return list(filter(filter_, marks))

    @abc.abstractmethod
    def __marks__(self, *series, **params):
        return []

    def _plot(self, **kwargs):
        kwargs.setdefault('y', self.series[0].data.mean())
        ax = self._digest.plot(**kwargs)
        self.series[0].plot(ax=ax)


@archetype
class HighlightSurvey(Survey):
    highlightertype = typeattribute(validate=fun.subcheck(Highlighter))

    def __call__(self, plot=False):
        """Calls the survey function with optional runtime arguments.

        The optional plot argument overlays the digest on the primary
        series before returning. The argument can take a dictionary whose
        key-value pairs will be passed to the plotting function.
        """
        highlights = self.highlights()
        digest = HighlightDigest.from_highlights(self.dataobject,
                                                 self.highlightertype,
                                                 highlights)
        self._digest = digest
        if plot is not False:
            if isinstance(plot, Mapping):
                self._plot(**plot)
            else:
                self._plot()
        return digest

    def highlight(self, shade, lower, upper):
        highlighter = self.highlightertype(shade)
        return Highlight(self.dataobject, highlighter, lower, upper)

    def highlights(self, *shades):
        """Returns the highlights from survey, with optional filter."""
        series = [s.data if isinstance(s, NxSeries) else s
                  for s in self.series]
        if any(s.empty for s in series) and not self.survey_empty:
            highlights = []
        else:
            highlights = self.__highlights__(*series, **self.params)

        def filter_(highlight):
            if not shades:
                return True
            else:
                return highlight.shade in shades
        return list(filter(filter_, highlights))

    @abc.abstractmethod
    def __highlights__(self, *series, **params):
        return []

    def _plot(self, **kwargs):
        if len(self.series[0]) <= 1:
            return
        if not isinstance(self.series[0], NxSeries):
            return
        mean = self.series[0].data.mean()
        std = self.series[0].data.std()
        y0, y1 = mean - 0.5 * std, mean + 0.5 * std
        kwargs.setdefault('y0', y0)
        kwargs.setdefault('y1', y1)
        ax = self._digest.plot(**kwargs)
        self.series[0].plot(ax=ax)
