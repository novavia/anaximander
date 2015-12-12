#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to the Anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

__all__ = ['utilities', 'types', 'Type', 'Object']


from . import utilities
from . import types
from .types import NxType as Type, NxObject as Object
