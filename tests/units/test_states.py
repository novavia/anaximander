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
from anaximander2.states import State

# =============================================================================
# Tests
# =============================================================================


class A_State(State):
    label = 'A'


class B_State(State):
    label = 'B'


@archetype
class Activity(State):
    pass


class Idle(Activity):
    label = 'idle'


class Operating(Activity):
    label = 'operating'


class Churning(Operating):
    label = 'churning'


class Activity_A(Activity):
    label = 'A'


def test_states():
    assert State.sublabels == ['A', 'B']
    assert Activity.sublabels == ['idle', 'operating', 'churning', 'A']
    op0 = Operating()
    op1 = Operating()
    assert op0 == op1
    s = {op0, op1}
    assert len(s) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
