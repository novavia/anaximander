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

from anaximander2.states import NxState, statetype, StateTransition
from anaximander2.events import HardEvent, SoftEvent

# =============================================================================
# Tests
# =============================================================================


class A_State(NxState):
    label = 'A'


class B_State(NxState):
    label = 'B'


@statetype
class Activity(NxState):
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
    assert NxState.sublabels == ['A', 'B']
    assert Activity.sublabels == ['null', 'idle', 'operating', 'churning', 'A']


def test_transitions():
    now = datetime.now()
    transition = Activity.Transition[Operating](now)
    assert isinstance(transition, HardEvent)
    assert isinstance(transition.state, Operating)
    transition = Churning.transition(now)
    assert isinstance(transition, StateTransition)
    assert isinstance(transition.state, Operating)
    assert transition.label == 'churning'


def test_phase():
    now = datetime.now()
    then = now + timedelta(60)
    phase = Activity.Phase[Operating](now, then)
    assert isinstance(phase, SoftEvent)
    assert isinstance(phase.state, Operating)
    phase = Activity.phase(now, then, label='idle')
    assert phase.label == 'idle'


def test_status():
    now = datetime.now()
    status = Activity.Status[Churning](now)
    assert isinstance(status, Activity.Status[Operating])
    assert isinstance(status.state, Churning)
    assert status.label == 'churning'
    status = Churning.status(now)
    assert status.label == 'churning'


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
