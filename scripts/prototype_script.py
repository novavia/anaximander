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

from anaximander.meta.metadescriptors import MetaCharacter
from anaximander.meta.nxmeta import ArcheType
from anaximander.meta.nxtype import NxType, prototype
from anaximander.meta.nxobject import NxObject

# =============================================================================
# Script
# =============================================================================


@prototype
class Object(NxObject):
    physical = MetaCharacter()


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
