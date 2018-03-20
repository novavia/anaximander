#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the base observation class.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Import statements
# =============================================================================

from abc import abstractproperty

from .utilities import functions as fun, xprops
from .structures import NxStructure

__all__ = ['NxObservation']

# =============================================================================
# Structure base class
# =============================================================================


class NxObservation(NxStructure):
    """Base class for observations.

    Observations are bundles of data and metadata that target a given object,
    and are typically localized in time and/or space. Relatively obvious
    examples include sampled values provided by a sensor, or the description of
    an event such as a weather phenomenon.

    Specifically, observations feature the following attributes:
        * object: the object of the observation, which could be an entity, a
            collection thereof, or really anything you fancy. An alternative
            name could have been 'context', but 'object' cements the notion
            that the observation applies to a tangible target. The attribute is
            implemented as a weakproperty such that observations don't persist
            if their object is disposed of.
        * locus: a property that localizes the observation, typically in space
            and/or time, but other dimensions (e.g. session identifier) are
            possible as well. The specifics of what is included in the locus
            are specified by the observation type, through an abstract
            property.
        * params: data and metadata that describes the observation, which
            is type-specific.
    """

    def __init__(self, object=None, **params):
        self.object = object
        self.params = params

    @xprops.weakproperty
    def object(self):
        return None

    @abstractproperty
    def locus(self):
        return None

    def __repr__(self):
        return fun.iformat('object', 'locus')(self)
