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

from anaximander.meta import metacharacter, prototype, NxObject
from anaximander.meta.nxmeta import ArcheType
from anaximander.meta.nxtype import NxType

# =============================================================================
# Script
# =============================================================================


@prototype
class Object(NxObject):
    physical = metacharacter()


class Hammer(Object):
    physical = True


class Noise(Object):
    physical = False


SpaceHammer = type(Hammer)('OtherHammer', (Object,), {}, physical='maybe')

if __name__ == '__main__':
    hammer = Hammer()
    assert isinstance(hammer, Object)
    assert issubclass(type(Object), ArcheType)
    assert isinstance(Object, NxType)
