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
        assert i1 not in interval

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


class TestMultiInterval(TestCase):

    def test_normalize(self):
        i0 = rge.FloatInterval(0, 1)
        i1 = rge.FloatInterval(0.5, 1.5)
        i2 = rge.FloatInterval(2, 3)
        i3 = rge.FloatInterval(0.25, 2.5)
        i4 = rge.FloatInterval(2.5, 3.5)
        i5 = rge.FloatInterval(0, 4)
        r0 = rge.FloatInterval(0, 1.5)
        r1 = rge.FloatInterval(0, 3)
        r2 = rge.FloatInterval(0, 3.5)
        assert rge.MultiInterval.normalize(i0) == [i0]
        assert rge.MultiInterval.normalize(i0, i1) == [r0]
        assert rge.MultiInterval.normalize(i0, i1, i2) == [r0, i2]
        assert rge.MultiInterval.normalize(i0, i1, i2, i3) == [r1]
        assert rge.MultiInterval.normalize(i0, i1, i2, i3, i4) == [r2]
        assert rge.MultiInterval.normalize(i5, i0, i1, i2, i3, i4) == [i5]

    def test_intersection(self):
        i0 = rge.FloatInterval(0, 1)
        i1 = rge.FloatInterval(0.5, 1.5)
        i2 = rge.FloatInterval(2, 3)
        i3 = rge.FloatInterval(0.25, 4)
        i4 = rge.FloatInterval(3.5, 5)
        r0 = rge.FloatInterval(0.5, 1)
        r1 = rge.FloatInterval(0.25, 1)
        r2 = rge.FloatInterval(3.5, 4)
        m1 = rge.MultiInterval([i0, i4])
        m2 = rge.MultiInterval([i2])
        assert rge.MultiInterval.intersection(m1, m2) == rge.MultiInterval()
        m2 = rge.MultiInterval([i1])
        assert rge.MultiInterval.intersection(m1, m2) == \
            rge.MultiInterval([r0])
        m2 = rge.MultiInterval([i3])
        assert rge.MultiInterval.intersection(m1, m2) == \
            rge.MultiInterval([r1, r2])

    def test_difference(self):
        i0 = rge.FloatInterval(0, 1)
        i1 = rge.FloatInterval(0.5, 1.5)
        i2 = rge.FloatInterval(2, 3)
        i3 = rge.FloatInterval(0.25, 5)
        i4 = rge.FloatInterval(3.5, 5)
        r0 = rge.FloatInterval(0, 0.5)
        r1 = rge.FloatInterval(0, 0.25)
        m1 = rge.MultiInterval([i0, i4])
        m2 = rge.MultiInterval([i2])
        assert rge.MultiInterval.difference(m1, m2) == m1
        m2 = rge.MultiInterval([i1])
        assert rge.MultiInterval.difference(m1, m2) == \
            rge.MultiInterval([r0, i4])
        m2 = rge.MultiInterval([i3])
        assert rge.MultiInterval.difference(m1, m2) == \
            rge.MultiInterval([r1])


def test_multi_time_interval():
    t0 = '2018-4-3 12:00'
    t1 = '2018-4-3 12:10'
    t2 = '2018-4-3 11:50'
    t3 = '2018-4-3 12:05'
    t4 = '2018-4-3 12:15'
    t5 = '2018-4-3 12:20'
    i0 = rge.TimeInterval(t0, t1)
    i1 = rge.TimeInterval(t2, t3)
    i2 = rge.TimeInterval(t4, t5)
    m = rge.MultiTimeInterval([i0, i1, i2])
    assert m.duration == pd.Timedelta('25m')
    assert m.gapspace == pd.Timedelta('5m')
    assert m.compact == rge.TimeInterval(t2, t5)


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
