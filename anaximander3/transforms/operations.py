#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines basic data operators.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc
import copy

import pandas as pd

from ..utilities import xprops, functions as fun, nxrange as rge
from ..data import datalogs as dtl
from . import logger as LOGGER
from .exceptions import InputError

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class Operation(abc.ABC):
    __inputs__ = ()
    __output__ = None
    __params__ = {}

    def __init__(self, *inputs, logger=LOGGER, **params):
        try:
            assert all(isinstance(i, t)
                       for i, t in zip(inputs, self.__inputs__))
        except AssertionError:
            itypes = tuple([t.__name__ for t in self.__inputs__])
            msg = f"Invalid inputs {inputs} to {type(self).__name__}, " + \
                  f"which expects types {itypes}."
            raise InputError(msg)
        self._inputs = inputs
        self.logger = logger
        self._params = copy.deepcopy(self.__params__)
        self._params.update(params)

    @xprops.cachedproperty
    def inputs(self):
        return []

    @property
    def input(self):
        """Shortcut to first input."""
        try:
            return self._inputs[0]
        except (AttributeError, IndexError):
            return None

    @xprops.cachedproperty
    def output(self):
        return None

    @property
    def params(self):
        return self._params.copy()

    @abc.abstractmethod
    def __operate__(self):
        """Type-specific operation method."""
        return None

    def __call__(self, plot=False):
        """Run interface."""
        try:
            output = self.__operate__()
            assert isinstance(output, self.__output__)
        except:
            if self.logger is None:
                raise
            msg = f"{type(self).__name__} operation error on {self.inputs}."
            self.logger.exception(msg, exc_info=True)
            return None
        self._output = output
        if plot:
            self.plot()
        return output

    def plot(self, **kwargs):
        if self.output is None:
            self()
        self.__plot__(**kwargs)

    def __plot__(self, **kwargs):
        """Optional operator plot."""
        pass


class Identity(Operation):
    __inputs__ = (dtl.DataLogsBase,)
    __output__ = dtl.DataLogsBase

    def __operate__(self):
        return self.inputs[0]


class LoggerIdentity(Operation):
    __inputs__ = (dtl.DataLogsBase,)
    __output__ = dtl.DataLogsBase

    def __operate__(self):
        msg = f"Processing {self.inputs[0]} in {type(self).__name__}."
        self.logger.debug(msg)
        return self.inputs[0]


class Thresholder(Operation):
    __inputs__ = (dtl.SampleSequence,)
    __output__ = dtl.EventSequence
    __params__ = {'column': None, 'threshold': 0}

    def __operate__(self):
        column, threshold = self.params['column'], self.params['threshold']
        df = self.input.data[[column]]
        df = df[df[column] >= threshold]
        df['label'] = 'peaking'
        return dtl.EventSequence(df[['label']], **self.input.metadata)

    def __plot__(self, **kwargs):
        score = self.input.plot(columns=[self.params['column']])
        staff = score.staves[0]
        staff.ax.axhline(self.params['threshold'], color='red')
        self.output.plot(staff=staff)


class MultiThresholder(Operation):
    __inputs__ = (dtl.SampleSequence,)
    __output__ = dtl.MultiEventSequence
    __params__ = {'columns': None, 'thresholds': {}}

    def __operate__(self):
        columns = fun.get(self.params['columns'], self.input.features)
        df = self.input.data[columns]
        event_series = []
        for col in df.columns:
            import pdb; pdb.set_trace()
            label = 'peaking_' + col
            index = df[df[col] >= self.params['thresholds'].get(col, 0)].index
            event_series.append(pd.DataFrame({'label': label}, index=index))
        return dtl.MultiEventSequence(pd.concat(event_series),
                                      **self.input.metadata)

    def __plot__(self, **kwargs):
        columns = fun.get(self.params['columns'], self.input.features)
        score = self.input.plot(columns=columns, certificate=False)
        for c, staff in zip(columns, score.staves):
            staff.ax.axhline(self.params['thresholds'].get(c, 0), color='red')
            label = 'peaking_' + c
            events = self.output.event_sequence(label)
            events.plot(staff=staff, make_main=True)


def sessionize(event_times, max_gap='0s', min_span='0s', label='label',
               onset_span=None):
    """Primitive for Sessionizer and MultiSessionizer."""
    max_gap = pd.Timedelta(max_gap)
    min_span = pd.Timedelta(min_span)
    # Extract the timestamp series of the events
    timestamps = pd.Series(event_times, index=event_times)
    # Compute consecutive differences and mark gaps
    gaps = timestamps.diff() > max_gap
    # The cumulative sum of gaps provide clusters of events
    clusters = gaps.cumsum()
    groups = clusters.groupby(clusters).groups
    # Session spans provided by first and last index of each group
    spans = [rge.TimeInterval(g[0], g[-1]) for g in groups.values()]
    # Check for onset session, and merge it if necessary
    onset_span = rge.time_range(onset_span)
    spans.append(onset_span)
    # Normalize spans
    spans = rge.MultiTimeInterval(spans, tolerance=max_gap).intervals
    # Extract qualified spans based on min_span
    qspans = [s for s in spans if s.duration >= min_span]
    # Turn spans into state transitions
    return pd.DataFrame({'datetime': [s.lower for s in qspans],
                         'duration': [s.duration for s in qspans],
                         'label': label})


class Sessionizer(Operation):
    __inputs__ = (dtl.EventSequence,)
    __output__ = dtl.SessionSequence
    # Params takes an optional onset session time interval
    __params__ = {'label': 'session', 'max_gap': '0s', 'min_span': '0s',
                  'onset_span': None}

    def __operate__(self):
        sessions_data = sessionize(self.input.datetime, **self.params)
        sessions = dtl.SessionSequence(sessions_data, **self.input.metadata)
        max_gap = pd.Timedelta(self.params['max_gap'])
        input_certif = self.input.certification
        if not pd.isna(input_certif):
            band = sessions[input_certif - max_gap, input_certif]
            if not band.empty:
                sessions.certify(band.datetime[0])
        return sessions

    def __plot__(self, **kwargs):
        score = self.input.plot(certificate=False)
        staff = score.staves[0]
        self.output.plot(staff=staff, make_main=True)


class MultiSessionizer(Operation):
    __inputs__ = (dtl.MultiEventSequence,)
    __output__ = dtl.MultiSessionSequence
    # Params takes an optional onset session
    __params__ = {'max_gap': '0s', 'min_span': '0s', 'labels': {},
                  'onset_spans': {}}

    def __operate__(self):
        sessions_ = []
        max_gap = pd.Timedelta(self.params['max_gap'])
        min_span = pd.Timedelta(self.params['min_span'])
        for label, group in self.input.data.groupby('label'):
            label = self.params['labels'].get(label, label)
            onset_span = self.params['onset_spans'].get(label)
            sessions_.append(sessionize(group.index.get_level_values(0),
                                        max_gap=max_gap, min_span=min_span,
                                        label=label, onset_span=onset_span))
        sessions_ = pd.concat(sessions_)
        metadata = self.input.metadata
        sessions = dtl.MultiSessionSequence(sessions_, **metadata)
        input_certif = self.input.certification
        if not pd.isna(input_certif):
            band = sessions[input_certif - max_gap, input_certif]
            if not band.empty:
                sessions.certify(min(band.datetime))
        return sessions

    def __plot__(self, **kwargs):
        score = self.input.plot(certificate=False)
        staff = score.staves[0]
        self.output.plot(staff=staff, make_main=True)
