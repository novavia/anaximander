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
        interval = rge.time_interval(lower, upper)
        assert interval.lower == datetime('2017-3-20')
        interval = rge.time_interval('2017-3-20 12:00', None)
        assert interval.upper == datetime.max
        with pytest.raises(ValueError):
            interval = rge.time_interval('one', 'two')
            interval.bounds
        assert rge.time_interval(interval) == interval

    def test_properties(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.time_interval(lower, upper)
        assert interval.length == pd.Timedelta(days=1)
        assert datetime('2017-3-20 12:00') in interval
        i0 = rge.time_interval('2017-3-20 12:00', '2017-3-20 13:00')
        assert i0 in interval
        i1 = rge.time_interval('2017-3-20 12:00', None)
        assert not i1 in interval        

    def test_sql(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.time_interval(lower, upper)
        sql = "timestamp >= '2017-03-20 00:00:00+00:00' AND " + \
              "timestamp <= '2017-03-21 00:00:00+00:00'"
        assert interval.sql('timestamp') == sql
        interval = rge.time_interval(lower)
        sql = "timestamp >= '2017-03-20 00:00:00+00:00'"
        assert interval.sql('timestamp') == sql
        interval = rge.time_interval()
        assert interval.sql('timestamp') == ""
        lower = 0.0
        upper = 1.0
        interval = rge.float_interval(lower, upper)
        sql = "x >= 0.0 AND x <= 1.0"
        assert interval.sql('x') == sql

class TestDiscreteRange(TestCase):

    def test_helper(self):
        item, items = 'item', ['i0', 'i1', 'i2']
        assert isinstance(rge.levels(item), rge.Level)
        assert isinstance(rge.levels(items), rge.Levels)
        
    def test_level(self):
        level = rge.Level('item')
        assert repr(level) == "Level('item')"
        assert level == 'item'

    def test_levels(self):
        levels = rge.Levels(['i0', 'i1', 'i2'])
        assert len(levels) == 3
        assert 'i0' in levels
        levels_repr = repr(levels._levels)
        assert repr(levels) == "Levels({0})".format(levels_repr)

    def test_sql(self):
        levels = rge.Levels(['i0', 'i1'])
        sqls = ["x IN ('i0', 'i1')", "x IN ('i1', 'i0')"]
        assert levels.sql('x') in sqls
        level = rge.Level('item')
        sql = "x = 'item'"
        assert level.sql('x') == sql

if __name__ == '__main__':
    pytest.main([__file__])
