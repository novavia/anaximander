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

from ..utilities import xprops, functions as fun
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
        except:
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
        for c in df.columns:
            label = 'peaking_' + c
            index = df[df[c] >= self.params['thresholds'].get(c, 0)].index
            event_series.append(pd.DataFrame({'label': label}, index=index))
        return dtl.MultiEventSequence(pd.concat(event_series),
                                      **self.input.metadata)

    def __plot__(self, **kwargs):
        columns = fun.get(self.params['columns'], self.input.features)
        score = self.input.plot(columns=columns)
        for c, staff in zip(columns, score.staves):
            staff.ax.axhline(self.params['thresholds'].get(c, 0), color='red')
            label = 'peaking_' + c
            events = self.output.event_sequence(label)
            events.plot(staff=staff)
