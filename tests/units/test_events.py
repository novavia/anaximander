#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxtypes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander2.meta import archetype
from anaximander2.events import Event

# =============================================================================
# Tests
# =============================================================================


class A_Event(Event):
    label = 'A'


class B_Event(Event):
    label = 'B'


@archetype
class Party(Event):
    pass


class Concert(Party):
    label = 'concert'


class Jam(Party):
    label = 'jam'


class Party_A(Party):
    label = 'A'


def test_events():
    assert Event.sublabels == ['A', 'B']
    assert Party.sublabels == ['concert', 'jam', 'A']


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
