#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module description.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

### Import statements ###

from abc import ABCMeta
from collections import defaultdict
from weakref import WeakValueDictionary


### NxType ###

class NxType(ABCMeta):
    """The parent type to all Anaximander types."""
    pass


d = defaultdict(list)
