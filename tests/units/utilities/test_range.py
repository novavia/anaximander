#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for functions.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

from unittest import TestCase

import pandas as pd
import pytest

from anaximander.utilities.nxtime import datetime
from anaximander.utilities import nxrange as rge

# =============================================================================
# Test Cases
# =============================================================================


class TestTimeInterval(TestCase):

    def test_init(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.TimeInterval(lower, upper)
        assert interval.lower == datetime('2017-3-20')
        interval = rge.TimeInterval('2017-3-20 12:00', None)
        assert interval.upper == datetime.max
        with pytest.raises(ValueError):
            rge.TimeInterval('one', 'two')

    def test_properties(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.TimeInterval(lower, upper)
        assert interval.length == pd.Timedelta(days=1)
        assert datetime('2017-3-20 12:00') in interval
        i0 = rge.TimeInterval('2017-3-20 12:00', '2017-3-20 13:00')
        assert i0 in interval
        i1 = rge.TimeInterval('2017-3-20 12:00', None)
        assert not i1 in interval        


class TestDiscreteRange(TestCase):

    def test_new(self):
        item, items = 'item', ['i0', 'i1', 'i2']
        assert isinstance(rge.DiscreteRange(item), rge.Level)
        assert isinstance(rge.DiscreteRange(items), rge.Levels)
        
    def test_level(self):
        level = rge.Level('item')
        assert repr(level) == "Level('item')"

    def test_levels(self):
        levels = rge.Levels(['i0', 'i1', 'i2'])
        assert len(levels) == 3
        assert 'i0' in levels
        levels_repr = repr(levels._levels)
        assert repr(levels) == "Levels({0})".format(levels_repr)

if __name__ == '__main__':
    pytest.main([__file__])
