#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for data.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander.data.states import State

# =============================================================================
# Test Cases
# =============================================================================


class MachineActivityState(State, abstract=True):
    pass


class OperatingState(MachineActivityState):
    label = 'operating'
    color = 'green'


class IdleState(MachineActivityState):
    label = 'idle'
    color = 'blue'


class SuperIdleState(IdleState):
    label = 'super_idle'
    color = 'white'


class PressActivityState(MachineActivityState, abstract=True):
    pass


class SingleStrokeOperatingState(PressActivityState):
    label = 'single_stroke'
    color = 'green'


def test_states():
    assert MachineActivityState.labels == {'operating', 'idle', 'undefined',
                                           'super_idle'}
    assert PressActivityState.labels == {'operating', 'idle', 'single_stroke',
                                         'undefined', 'super_idle'}
    assert IdleState() == IdleState()
    assert OperatingState != IdleState()
    idle_0, idle_1 = IdleState(), IdleState()
    idle_0.comment = "no power"
    assert idle_0 != idle_1
    with pytest.raises(TypeError):
        class IdlingState(IdleState, abstract=True):
            pass
    with pytest.raises(TypeError):
        class PressMachineActivityState(PressActivityState,
                                        MachineActivityState, abstract=True):
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
