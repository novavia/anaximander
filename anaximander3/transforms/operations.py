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

from ..utilities import xprops
from ..data.datalogs import DataLogsBase
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
        self._params = self.__params__.copy()
        self._params.update(params)

    @xprops.cachedproperty
    def inputs(self):
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

    def __call__(self):
        """Run interface."""
        output = self.__operate__()
        self._output = output
        return output


class Identity(Operation):
    __inputs__ = (DataLogsBase,)
    __output__ = DataLogsBase

    def __operate__(self):
        return self.inputs[0]


class LoggerIdentity(Operation):
    __inputs__ = (DataLogsBase,)
    __output__ = DataLogsBase

    def __operate__(self):
        msg = f"Processing {self.inputs[0]} in {type(self).__name__}."
        self.logger.debug(msg)
        return self.inputs[0]
