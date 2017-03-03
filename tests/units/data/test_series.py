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
import pandas as pd
import pytest

from anaximander.data import quantities as qnt, NxScalar, NxSeries

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
    SpeedMPH, _ = datatypes
    assert issubclass(NxSeries[SpeedMPH], NxSeries)
    assert NxSeries[SpeedMPH].unit == 'mph'


def test_instance(datatypes):
    SpeedMPH, _ = datatypes
    smph = NxSeries[SpeedMPH]([25, 35, 55, 65])
    assert isinstance(smph.data, pd.Series)
    assert smph.data.data == np.array([25, 35, 55, 65])
    assert repr(smph) == '<NxSeries[SpeedMPH](4 items)>'


def test_conversion(datatypes):
    SpeedMPH, SpeedKPH = datatypes
    smph = NxSeries[SpeedMPH]([25, 35, 55, 65])
    skph = smph.convert(SpeedKPH)
    assert type(skph) is NxSeries[SpeedKPH]
    assert skph.data[0] == 25 / 0.621371


def test_indexing(datatypes):
    SpeedMPH, _ = datatypes
    smph = NxSeries[SpeedMPH]([25, 35, 55, 65])
    assert smph[0] == SpeedMPH(25)
    assert type(smph[0:2]) is NxSeries[SpeedMPH]
    assert len(smph[0:2]) == 2
    assert type(smph.loc[0:2]) is NxSeries[SpeedMPH]
    assert len(smph.loc[0:2]) == 3
    assert type(smph[smph.data.values > 50]) is NxSeries[SpeedMPH]
    assert smph[smph.data.values > 50].data.data == np.array([55, 65])


def test_extend(datatypes):
    SpeedMPH, _ = datatypes
    smph = NxSeries[SpeedMPH]([25, 35, 55, 65])
    smph.extend(smph.data)
    assert len(smph) == 8


if __name__ == '__main__':
    pytest.main([__file__])
