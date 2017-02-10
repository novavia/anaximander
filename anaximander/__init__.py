#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to the Anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

import os.path
import sys

from . import utilities
from . import registries
from . import meta
from . import data


# Sets the anaximander directory
NXDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Appends the test directory to the import path
TESTDIR = os.path.join(NXDIR, 'tests')
sys.path.append(TESTDIR)

__all__ = ['utilities', 'registries', 'meta', 'data', 'NXDIR']
