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
from anaximander.meta.nxtype import directory
from anaximander.meta.nxobject import NxObject

# =============================================================================
# Quantity class
# =============================================================================


@nxattr.s
class Quantity(NxObject, registry=directory('name')):
    """A class that holds a name and a unit."""
    name = nxattr.ib(validator=nxattr.validators.instance_of(str))
    unit = nxattr.ib(validator=nxattr.validators.instance_of(str))

    def __attrs_post_init__(self):
        """Adds instances to the module's dictionary so they are persisted."""
        globals()[self.name] = self
