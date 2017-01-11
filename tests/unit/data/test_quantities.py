#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for quantities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander.data import quantities as qnt

# =============================================================================
# Test Cases
# =============================================================================


def test_instantiation():
    qnt.Quantity('speed', 'mph')
    assert qnt.Quantity['speed'].measure == 'mph'
    assert qnt.speed == qnt.Quantity['speed']
    qnt.Quantity('speed', 'kph')
    assert qnt.Quantity['speed'].measure == 'kph'
    assert qnt.speed == qnt.Quantity['speed']
    with pytest.raises(TypeError):
        qnt.Quantity(0, 0)


@pytest.fixture
def quantities():
    speed = qnt.Quantity('speed', 'mph')
    speed.register_unit('kph', 0.621371)
    temperature = qnt.Quantity('temperature', 'K')
    zero = 273.15
    temperature.register_unit('C', lambda t: t + zero, lambda t: t - zero)
    temperature.register_unit('F', lambda t: (t + 459.67) * 5 / 9,
                              lambda t: t * 9/5 - 459.67)
    return speed, temperature


def test_unit_registration(quantities):
    speed, temperature = quantities
    assert speed.units == {'mph': 1., 'kph': 0.621371}
    temperature.deregister_unit('F')
    assert set(temperature.units) == {'K', 'C', '_C'}
    with pytest.raises(qnt.UnitError):
        # Missing reverse conversion function.
        temperature.register_unit('F', lambda t: (t + 459.67) * 5 / 9)


def test_unit_conversion(quantities):
    speed, temperature = quantities

    smph = 65
    skph = speed.convert(smph, target='kph')
    assert skph == pytest.approx(smph / 0.621371)
    assert speed.convert(skph, 'kph') == pytest.approx(smph)
    tfar = 69
    tcel = temperature.convert(tfar, 'F', 'C')
    assert tcel == pytest.approx(20.555556)
    assert temperature.convert(tcel, 'C', 'F') == pytest.approx(tfar)


def test_bulk_unit_conversion(quantities):
    speed, temperature = quantities
    smph = [35, 55, 75]
    skph = list(speed.iterconvert(smph, target='kph'))
    assert skph == pytest.approx([v / 0.621371 for v in smph])
    assert list(speed.iterconvert(skph, 'kph')) == pytest.approx(smph)
    tfar = [69, 100]
    tcel = list(temperature.iterconvert(tfar, 'F', 'C'))
    assert tcel == pytest.approx([20.555556, 37.777778])
    assert list(temperature.iterconvert(tcel, 'C', 'F')) == pytest.approx(tfar)


if __name__ == '__main__':
    pytest.main([__file__])
