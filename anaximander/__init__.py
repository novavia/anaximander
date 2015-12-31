#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to the Anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

__all__ = ['nxtype', 'nxobject', 'utilities', '_meta', 'Type', 'Object']


from ._arche import nxtype, nxobject
from . import utilities
from . import _meta
from .types import Type, Object
