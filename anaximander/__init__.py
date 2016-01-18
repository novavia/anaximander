#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to the Anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

__all__ = ['utilities', 'nxtype', 'nxobject', 'Type', 'Object']


from . import utilities
from ._arche import nxtype, nxobject
from .types import Type, Object
