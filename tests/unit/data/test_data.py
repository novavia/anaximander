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

from anaximander.data import quantities as qnt
from anaximander.data.data import NxData

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def datatypes():

    speed = qnt.Quantity('speed', 'mph')
    speed.register_unit('kph', 0.621371)

    class SpeedMPH(NxData):
        quantity = speed

    class SpeedKPH(NxData):
        quantity = speed
        unit = 'kph'

    return SpeedMPH, SpeedKPH


def test_type_creation(datatypes):
    SpeedMPH, SpeedKPH = datatypes
    assert SpeedMPH.quantity == SpeedKPH.quantity == qnt.speed
    assert SpeedMPH.unit == 'mph'
    assert SpeedKPH.unit == 'kph'


if __name__ == '__main__':
    pytest.main([__file__])
