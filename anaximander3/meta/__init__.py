#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to Anaximander's meta package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class NxMetaError(Exception):
    """Customized error for metaprogramming errors."""
    pass


from .nxdescriptors import *
