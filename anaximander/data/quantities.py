#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantities module, which defines a class for physical quantities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

from anaximander.utilities import nxattr

# =============================================================================
# Quantity class
# =============================================================================


def isstring(s):
    """Returns True if s is a string, False otherwise."""
    return isinstance(s, str)


@nxattr.s
class Quantity:
    """A class that holds a name and a unit."""
    name = nxattr.ib(validator=isstring)
    unit = nxattr.ib(validator=isstring)
