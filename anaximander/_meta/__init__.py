#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to Anaximander's _meta package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

__all__ = ['Meta', 'BaseType', 'BaseObject', 'nxframe']

from .nxmeta import NxMeta as Meta, \
    NxBaseType as BaseType, \
    NxBaseObject as BaseObject
from . import nxframe
