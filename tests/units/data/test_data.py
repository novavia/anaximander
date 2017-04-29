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

import numpy as np
import pytest

from anaximander.data import quantities as qnt, NxScalar

# =============================================================================
# Test Cases
# =============================================================================


speed = qnt.Quantity('speed', 'mph')
speed.register_unit('kph', 0.621371)


@pytest.fixture
def datatypes():

    class SpeedMPH(NxScalar):
        quantity = speed

    class SpeedKPH(NxScalar):
        quantity = speed
        unit = 'kph'
        precision = 2

    return SpeedMPH, SpeedKPH


def test_type_creation(datatypes):
    SpeedMPH, SpeedKPH = datatypes
    assert SpeedMPH.quantity == SpeedKPH.quantity
    assert SpeedMPH.unit == 'mph'
    assert SpeedKPH.unit == 'kph'


def test_instance(datatypes):
    SpeedMPH, SpeedKPH = datatypes
    smph = SpeedMPH(np.float(65))
    skph = SpeedKPH(np.float(100))
    assert smph.data == np.float(65)
    assert smph.context is None
    assert repr(smph) == 'SpeedMPH(65.0)'
    assert str(smph) == '65.0 mph'
    assert repr(skph) == "SpeedKPH(100.0)"
    assert str(skph) == '1e+02 kph'


def test_comparisons(datatypes):
    SpeedMPH, SpeedKPH = datatypes
    assert SpeedMPH(65) == SpeedMPH(65)
    assert SpeedKPH(100.005) == SpeedKPH(100)
    assert not SpeedMPH(65.005) == SpeedMPH(65)
    # This is a bit of a corner case, because precision is defined
    # in SpeedKPH but not SpeedMPH. So this makes sense from the
    # standpoint of the API, but that's obviously weird and error-prone...
    assert SpeedKPH(100) == SpeedMPH(62.14)
    assert SpeedMPH(62.14) != SpeedKPH(100)
    assert SpeedMPH(62.14) > SpeedKPH(100)
    assert SpeedKPH(100) <= SpeedMPH(62.14)
    assert not SpeedKPH(100) < SpeedMPH(62.14)
    assert SpeedMPH(62.1) < SpeedKPH(100)
    assert SpeedKPH(100) >= SpeedMPH(62.1)
    assert SpeedKPH(100) > SpeedMPH(62.1)
    assert not SpeedKPH(100) > SpeedMPH(62.13)


def test_conversion(datatypes):
    SpeedMPH, SpeedKPH = datatypes
    speed_mph = SpeedKPH(100).convert(SpeedMPH)
    assert type(speed_mph) is SpeedMPH
    assert speed_mph == SpeedMPH(62.1371)


if __name__ == '__main__':
    pytest.main([__file__])
