#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimental module to redefine base sequence types.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc
import bisect
from collections import defaultdict
from itertools import chain

import pandas as pd

import anaximander as nx
from ..utilities import functions as fun, nxrange as rng, nxattr, nxtime
from .data import NxFloat
from .fields import Str, Scalar, Timestamp, Period
from .schema import Schema
from .frame import NxDataSequence
from .states import State, UndefinedState

if nx.INTERACTIVE:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

__all__ = []

# =============================================================================
# Schema classes
# =============================================================================


class TimeSequenceSchema(Schema):
    """Base class for schemas indexed with timestamps."""
    timestamp = Timestamp(key=True, sequential=True)


class SampleSequenceSchema(TimeSequenceSchema):
    samples = Scalar(NxFloat)


class EventSequenceSchema(TimeSequenceSchema):
    event = Str()


class PhaseSequenceSchema(TimeSequenceSchema):
    state = Str()


class StateSequenceSchema(TimeSequenceSchema):
    state = Str()


class PeriodSequenceSchema(TimeSequenceSchema):
    timestamp = Period(freq='H', key=True, sequential=True)
    summary = Scalar(NxFloat)


class TimeSequence(NxDataSequence, schema=TimeSequenceSchema):

    def __init__(self, data, context=None, validate=False,
                 lower=None, upper=None):
        # XXX: this won't work because it's passed to a time interval that
        # enforces lower to be less than upper.
        # lower = fun.get(lower, nxt.MAX)
        # upper = fun.get(upper, nxt.MIN)
        super().__init__(data, context, validate, timestamp=(lower, upper))


class SampleSequence(TimeSequence, schema=SampleSequenceSchema):
    pass


class EventSequence(TimeSequence, schema=EventSequenceSchema):
    pass


class PhaseSequence(TimeSequence, schema=PhaseSequenceSchema):

    def __init__(self, data, context=None, validate=False,
                 lower=None, upper=None, onstate=None):
        """The 'onstate' provides the begin state of the sequence.

        If the lower certificate is less than the smallest index of the
        sequence -including the case where the sequence is empty, the onstate
        provides the state at the lower certificate. Otherwise the onstate is
        irrelevant since the state of the sequence at the lower certificate
        can be determined from the sequence itself. There is still the
        exception of a case where the upper certificate is less than the
        lower certificate, in which case onstate is also irrelevant.
        """
        super().__init__(data, context, validate, lower, upper)
        self._onstate = onstate

    @property
    def onstate(self):
        return self._onstate

    def loc_state(self, timestamp):
        """Return the state at the specified timestamp."""
        timestamp = nxtime.datetime(timestamp)
        ix = bisect.bisect(self.data['timestamp'], timestamp) - 1
        if ix == -1:
            if self.lower is not None:
                if timestamp >= self.lower:
                    return self.onstate, True
            else:
                raise IndexError
        elif ix == len(self) - 1:
            if self.upper is not None:
                if timestamp <= self.upper:
                    return self.data['state'][-1], True
            else:
                raise IndexError
        state_ = self.data['state'][ix]
        if self.lower is None or self.upper is None:
            return state_, False
        else:
            if timestamp >= self.lower and timestamp <= self.upper:
                return state_, True
            else:
                return state_, False

    @property
    def phases(self):
        """Returns active phases."""
        df = self.data
        # If no transitions, either one or zero phases
        if df.empty:
            if self.lower is None or self.upper is None:
                return []
            else:
                phase = Phase(self.onstate,
                              rng.TimeInterval(self.lower, self.upper),
                              True)
                return [phase]
        # If there is a onstate, we add a transition to materialize it
        if self.onstate != 'blank' and self.lower is not None:
            if self.lower not in self.index:
                transition = pd.Series({'timestamp': self.lower,
                                        'state': self.onstate},
                                       name=self.lower)
                df.append(transition)
        # If the last transition is beyond upper, we also materialize that
        if self.upper is not None:
            if df.index[-1] > self.upper:
                upper_state = self.loc_state(self.upper)
                if upper_state != 'blank':
                    if self.upper not in self.index:
                        transition = pd.Series({'timestamp': self.upper,
                                                'state': upper_state},
                                               name=self.upper)
                        df.append(transition)
        df = df.sort_index()
        # We join timestamp on itself to form consecutive pairs
        nexts = df['timestamp'].shift(-1)
        # The upper certificate serves as upper bound if no transition succeeds
        if self.upper is not None:
            if df.index[-1] <= self.upper:
                nexts.iloc[-1] = self.upper
        df['next'] = nexts
        # Selects only the active phases
        active_phases = df[df.state != 'blank'][['state', 'timestamp', 'next']]
        # If the last active phase is unbounded, we remove it
        if not active_phases.empty:
            if active_phases['next'][-1] is pd.NaT:
                active_phases = active_phases.iloc[:-1]
        # If empty certificates, then there only uncertified phases
        if self.lower is None or self.upper is None:
            return [Phase(v[0], rng.TimeInterval(v[1], v[2]), False)
                    for v in active_phases.values]
        phases = [Phase(v[0], rng.TimeInterval(v[1], v[2]), True)
                  for v in active_phases.values]
        for p in phases:
            if p.lower < self.lower:
                p.certified = False
            elif p.lower >= self.upper:
                p.certified = False
        return phases


class StateSequence(TimeSequence, schema=StateSequenceSchema):

    def __init__(self, data, context=None, validate=False,
                 lower=None, upper=None, statetype=None, onstate=None):
        # XXX: onstate is a bit misguided because it should generally map
        # to the sequence's fields.
        # A field is also tied to a StateType so we can determine the
        # state from a string.
        """The 'onstate' provides the begin state of the sequence.

        Lower and upper should be validated. None means MIN for lower and
        MAX for upper.
        """
        super().__init__(data, context, validate, lower, upper)
        # XXX: technically we should normalize this to be either the
        # state instance or the string -but accept either option in the init
        if statetype is None:
            raise ValueError
        self._statetype = statetype
        if not isinstance(onstate, State):
            if onstate is None:
                onstate = UndefinedState()
            else:
                onstate = statetype(onstate)
        self._onstate = onstate

    @property
    def onstate(self):
        return self._onstate

    @property
    def statetype(self):
        return self._statetype

    def loc_state(self, timestamp):
        """Return the state at the specified timestamp."""
        timestamp = nxtime.datetime(timestamp)
        if timestamp < self.lower or timestamp > self.upper:
            raise IndexError
        ix = bisect.bisect(self.data['timestamp'], timestamp) - 1
        if ix == -1:
            return self.statetype(self.onstate)
        return self.statetype(self.data['state'][ix])

    def phases(self, *states):
        """Returns phases for supplied states or all states."""
        # We must materialize a virtual transition at the lower end
        lower = pd.DataFrame({'timestamp': self.lower,
                              'state': self.onstate.label},
                             index=[self.lower])
        if self.data.empty:
            df = lower
        else:
            df = lower.append(self.data)
        # We join timestamp on itself to form consecutive pairs
        nexts = df['timestamp'].shift(-1)
        nexts.iloc[-1] = self.upper
        df['next'] = nexts
        df = df[['state', 'timestamp', 'next']]
        if states:
            df = df[df.state.isin(states)]
        return [Phase(self.statetype(v[0]), rng.TimeInterval(v[1], v[2]))
                for v in df.values]


class PeriodSequence(TimeSequence, schema=PeriodSequenceSchema):
    pass


@nxattr.s
class Phase:
    state = nxattr.ib()
    interval = nxattr.ib()
    certified = nxattr.ib(default=True)  # T / F

    @property
    def lower(self):
        return self.interval.lower

    @property
    def upper(self):
        return self.interval.upper

    @property
    def length(self):
        return self.interval.length

    @property
    def bounds(self):
        return self.interval.bounds

    @property
    def color(self):
        return self.state.color


class MultiPhase:
    """A sequence of Phases in a single state."""

    def __init__(self, phases=None):
        self.phases = list(fun.get(phases, []))
        if not phases:
            self._state = None
        else:
            states = set(p.state for p in phases)
            if len(states) != 1:
                raise ValueError
            self._state = states.pop()

    @property
    def state(self):
        return self._state

    @property
    def multi_interval(self):
        return rng.MultiTimeInterval([p.interval for p in self.phases])


# =============================================================================
# Operators
# =============================================================================


class Operator:

    @abc.abstractmethod
    def __call__(self, plot=False, **kwargs):
        return NotImplemented

    @abc.abstractclassmethod
    def __plot__(self, output, **kwargs):
        return NotImplemented


class Thresholder(Operator):

    def __init__(self, sample_sequence, threshold=0):
        self.sample_sequence = sample_sequence
        self.threshold = threshold

    def __call__(self, plot=False, **kwargs):
        dataframe = self.sample_sequence.data
        events = dataframe[dataframe.samples >= self.threshold].copy()
        events['event'] = 'peaking'
        output = EventSequence(events[['timestamp', 'event']],
                               lower=self.sample_sequence.lower,
                               upper=self.sample_sequence.upper)
        if plot:
            self.__plot__(output)
        return output

    def __plot__(self, output, **kwargs):
        ax = plot_series(self.sample_sequence.samples())
        plot_marks(output, self.threshold, ax=ax)


class SessionMaker(Operator):

    class SessionState(State, abstract=True):
        pass

    class Session(SessionState):
        label = 'session'
        color = 'green'

    class Idle(SessionState):
        label = 'idle'
        color = 'white'

    def __init__(self, event_sequence, max_gap, min_span=None):
        self.event_sequence = event_sequence
        self.max_gap = pd.Timedelta(max_gap)
        if min_span is None:
            self.min_span = None
        else:
            self.min_span = pd.Timedelta(min_span)

    def __old_call__(self, plot=False, **kwargs):
        # Extract the timestamp series of the events
        events = self.event_sequence.data
        timestamps = pd.Series(events.timestamp, index=events.timestamp)
        # Compute consecutive differences and mark gaps
        gaps = timestamps.diff() > self.max_gap
        # The cumulative sum of gaps provide clusters of events
        clusters = gaps.cumsum()
        groups = clusters.groupby(clusters).groups
        # Session spans provided by first and last index of each group
        spans = [rng.TimeInterval(g[0], g[-1]) for g in groups.values()]
        # Extract qualified spans based on min_span
        if self.min_span is not None:
            qspans = [s for s in spans if s.duration >= self.min_span]
        else:
            qspans = spans
        # Turn spans into state transitions
        index = list(chain(*qspans))
        states = ['session', 'blank'] * len(qspans)
        transitions = pd.DataFrame({'timestamp': index, 'state': states},
                                   index=index)

        # Determine lower and upper certificates
        # We first assess events that are between certificate bounds
        events_lower = self.event_sequence.lower
        events_upper = self.event_sequence.upper
        onstate = 'blank'  # This a temporary assumption
        if events_lower < events_upper:
            certified_events = events.loc[events_lower:events_upper]
        else:
            certified_events = pd.DataFrame()
        # If no such events, we assess the gap between event certificates
        if certified_events.empty:
            if (events_upper - events_lower) < self.max_gap:
                lower = upper = None
            else:
                lower, upper = events_lower, events_upper
        # Otherwise we compare certificates with first (resp. last) event
        else:
            first_event = certified_events.timestamp[0]
            last_event = certified_events.timestamp[-1]
            # If there is a gap, we can certify the sequence at the onset
            if first_event - events_lower >= self.max_gap:
                lower = events_lower
            # Otherwise the situation depends on whether the first span
            # is qualified or not
            # If qualified, the certificate is the first event
            # We remove the first transitions and instead set the onstate
            # to the active state
            # XXX: DO NOT REMOVE TRANSITION
            elif spans[0] in qspans:  # if events exist, spans is not empty
                lower = first_event
                transitions = transitions.iloc[1:]
                onstate = 'session'
            # If not qualified, certificate is at the last event of the
            # first span
            else:
                lower = spans[0].upper
            if events_upper - last_event >= self.max_gap:
                upper = events_upper
            # XXX: DO NOT REMOVE TRANSITION
            elif spans[-1] in qspans:
                upper = last_event
                transitions = transitions.iloc[:-1]
            else:
                upper = spans[-1].lower

        output = PhaseSequence(transitions, lower=lower, upper=upper,
                               onstate=onstate)
        if plot:
            self.__plot__(output)
        return output

    def __call__(self, plot=False, **kwargs):
        # Extract the timestamp series of the events
        events = self.event_sequence.data
        events_lower = self.event_sequence.lower
        events_upper = self.event_sequence.upper
        timestamps = pd.Series(events.timestamp, index=events.timestamp)
        # Compute consecutive differences and mark gaps
        gaps = timestamps.diff() > self.max_gap
        # The cumulative sum of gaps provide clusters of events
        clusters = gaps.cumsum()
        groups = clusters.groupby(clusters).groups
        # Session spans provided by first and last index of each group
        spans = [rng.TimeInterval(g[0], g[-1]) for g in groups.values()]
        # Extract qualified spans based on min_span
        if self.min_span is not None:
            qspans = [s for s in spans if s.duration >= self.min_span]
        else:
            qspans = spans
        # Turn spans into state transitions
        index = list(chain(*qspans))
        states = ['session', 'idle'] * len(qspans)
        transitions = pd.DataFrame({'timestamp': index, 'state': states},
                                   index=index)

        onstate = 'idle'  # placeholder
        if events.empty:
            if events_upper - events_lower < self.max_gap:
                return None
            lower, upper = events_lower, events_upper
        else:
            first_event, last_event = timestamps[0], timestamps[-1]
            # If there is a gap, we can certify the sequence at the onset
            if first_event - events_lower >= self.max_gap:
                lower = events_lower
            # Otherwise the situation depends on whether the first span
            # is qualified or not
            # If qualified, the certificate is the first event
            # We remove the first transitions and instead set the onstate
            # to the active state
            elif spans[0] in qspans:  # if events exist, spans is not empty
                lower = first_event
                transitions = transitions.iloc[1:]
                onstate = 'session'
            # If not qualified, certificate is at the last event of the
            # first span provided it is further than max_gap from the events
            # upper bound.
            elif events_upper - spans[0].upper >= self.max_gap:
                lower = spans[0].upper
            else:
                return None
            if events_upper - last_event >= self.max_gap:
                upper = events_upper
            elif spans[-1] in qspans:
                upper = last_event
                transitions = transitions.iloc[:-1]
            elif spans[-1].lower - events_lower >= self.max_gap:
                upper = spans[-1].lower
            else:
                return None

        output = StateSequence(transitions, lower=lower, upper=upper,
                               onstate=onstate, statetype=self.SessionState)
        if plot:
            self.__plot__(output)
        return output

    def __plot__(self, output, **kwargs):
        if output is None:
            return
        statetype = output.statetype
        d = defaultdict(list)
        for p in output.phases():
            d[p.state.label].append(p)
        ax = None
        for state, phases in d.items():
            color = statetype(state).color
            if ax is None:
                ax = plot_highlights(phases, 90, 110, color=color)
            else:
                plot_highlights(phases, 90, 110, ax=ax, color=color)
        # XXX: for whatever reason, when there is a single active phase,
        # plotting the events makes the phases disappear... That does not
        # happen with a single overall phase or multiple active phases...
#        plot_marks(self.event_sequence, 100, ax=ax)

# =============================================================================
# Plotting functions
# =============================================================================


def plot_series(series, ax='new', **kwargs):
    """Customized plot function.

    Params:
        ax: 'new' for new plot, otherwise see pandas series plot.
    """
    if ax == 'new':
        fig, ax = plt.subplots()
    ax = series.data.plot(ax=ax, **kwargs)
    ax.set_xlabel(series.index.name)
    ax.set_ylabel(series.data.name)
    return ax


def plot_marks(events, y, ax='new', **kwargs):
    """Plots marks from events, at specified y-axis value."""
    if events.empty:
        return
    if ax == 'new':
        fig, ax = plt.subplots()
    plot_data = pd.Series(y, index=events.timestamp())
    plot_data.plot(ax=ax, marker='o', ls='none', color='red')
    return ax


def plot_highlights(phases, y0, y1, ax='new', **kwargs):
    """Plots highlights as shaded blocks between specified y-axis values."""
    if ax == 'new':
        fig, ax = plt.subplots()
    color = kwargs.get('color', 'red')
    intervals = [p.interval for p in phases]
    timestamps = (mdates.date2num(t) for t in chain(*intervals))
    intervals = [(p, n - p) for p, n in fun.pairwise(timestamps, 2)]
    ax.broken_barh(intervals, (y0, y1 - y0), facecolor=color, zorder=-1)
    return ax
