#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines jobs, intended as units of work.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements and constants
# =============================================================================

import abc

from . import logger as LOGGER
from .exceptions import InputError

__all__ = []

# =============================================================================
# Base type
# =============================================================================


class Job(abc.ABC):
    __etype__ = None

    def __init__(self, entity, input_store=None, output_store=None,
                 logger=LOGGER, **params):
        try:
            assert isinstance(entity, self.__etype__)
            assert hasattr(entity, 'id')
        except AssertionError:
            msg = f"Improper entity {entity} supplied to " + \
                  f"{type(self).__name__} job."
            raise InputError(msg)
        self.entity = entity
        self.input_store = input_store
        self.output_store = output_store
        self.logger = logger

    def retrieve_inputs(self):
        pass

    @abc.abstractmethod
    def __work__(self):
        """Type-specific work method."""
        return None

    def __call__(self, plot=False):
        """Run interface."""
        self.retrieve_inputs()
