#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxtype.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import abc

from anaximander.meta.nxtype import prototype
from anaximander.meta.nxobject import NxObject

# =============================================================================
# Script
# =============================================================================


@prototype
class Object(NxObject):

    @abc.abstractproperty
    def physical(self):
        return NotImplemented


class Hammer(Object):

    @property
    def physical(self):
        return True


class Noise(Object):

    @property
    def physical(self):
        return False


if __name__ == '__main__':
    hammer = Hammer()
#    import pdb; pdb.set_trace()
    assert isinstance(hammer, Object)
