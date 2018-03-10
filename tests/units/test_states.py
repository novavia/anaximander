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

from datetime import datetime, timedelta
import pytest

from anaximander2.meta import archetype
from anaximander2.states import State, statetype
from anaximander2.events import HardEvent, SoftEvent

# =============================================================================
# Tests
# =============================================================================


class A_State(State):
    label = 'A'


class B_State(State):
    label = 'B'


@statetype
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
    assert Activity.sublabels == ['null', 'idle', 'operating', 'churning', 'A']
    now = datetime.now()
    transition = Activity.Transition['operating'](now)
    phase = Activity.Phase['operating'](now, now + timedelta(60))
    assert isinstance(transition, HardEvent)
    assert isinstance(phase, SoftEvent)
    assert isinstance(transition.state, Operating)
    assert isinstance(phase.state, Operating)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
