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

import json
from unittest import TestCase

import pandas as pd
import pytest

from anaximander3.utilities.nxtime import datetime
from anaximander3.utilities import nxrange as rge

# =============================================================================
# Test Cases
# =============================================================================


class TestTimeInterval(TestCase):

    def test_init(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.time_range(lower, upper)
        assert interval.lower == datetime('2017-3-20')
        assert rge.time_range(slice(lower, upper)) == interval
        interval = rge.TimeInterval('2017-3-20 12:00')
        assert interval.upper == datetime.max
        with pytest.raises(ValueError):
            interval = rge.time_range('one', 'two')
            interval.bounds
        assert rge.time_range(interval) == interval

    def test_properties(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.time_range(lower, upper)
        assert interval.length == pd.Timedelta(days=1)
        assert datetime('2017-3-20 12:00') in interval
        i0 = rge.time_range('2017-3-20 12:00', '2017-3-20 13:00')
        assert i0 in interval
        i1 = rge.TimeInterval('2017-3-20 12:00')
        assert i1 not in interval

    def test_sql(self):
        lower = '2017-3-20'
        upper = '2017-3-21'
        interval = rge.time_range(lower, upper)
        sql = "timestamp >= '2017-03-20 00:00:00+00:00' AND " + \
              "timestamp < '2017-03-21 00:00:00+00:00'"
        assert interval.sql('timestamp') == sql
        interval = rge.TimeInterval(lower)
        sql = "timestamp >= '2017-03-20 00:00:00+00:00'"
        assert interval.sql('timestamp') == sql
        interval = rge.TimeInterval()
        assert interval.sql('timestamp') == ""
        lower = 0.0
        upper = 1.0
        interval = rge.float_range(lower, upper)
        sql = "x >= 0.0 AND x < 1.0"
        assert interval.sql('x') == sql

    def test_intersection(self):
        l0 = '2018-4-23 15:40'
        u0 = '2018-4-23 16:00'
        l1 = '2018-4-23 15:50'
        u1 = '2018-4-23 16:05'
        l2 = u0
        u2 = u1
        i0 = rge.time_range(l0, u0)
        i1 = rge.time_range(l1, u1)
        i2 = rge.time_range(l2, u2)
        t0 = rge.time_range(l0)
        assert i0 & i1 == rge.time_range(l1, u0)
        assert i0 & i2 == rge.EmptyTimeInterval()
        assert i0 & t0 == t0
        assert t0 & i0 == t0
        assert i1 & t0 == rge.EmptyTimeInterval()
        assert t0 & i1 == rge.EmptyTimeInterval()

    def test_union(self):
        l0 = '2018-4-23 15:40'
        u0 = '2018-4-23 15:45'
        l1 = '2018-4-23 15:50'
        u1 = '2018-4-23 16:05'
        l2 = u0
        u2 = u1
        i0 = rge.time_range(l0, u0)
        i1 = rge.time_range(l1, u1)
        i2 = rge.time_range(l2, u2)
        assert i0 | i1 == rge.MultiTimeInterval([i0, i1])
        assert i0 | i2 == rge.time_range(l0, u1)

    def test_compact(self):
        l0 = '2018-4-23 15:40'
        u0 = '2018-4-23 15:45'
        l1 = '2018-4-23 15:50'
        u1 = '2018-4-23 16:05'
        i0 = rge.time_range(l0, u0)
        i1 = rge.time_range(l1, u1)
        assert rge.TimeInterval.compact(i0, i1) == rge.time_range(l0, u1)

    def test_serialization(self):
        l0 = '2018-4-23 15:40'
        u0 = '2018-4-23 15:45'
        i0 = rge.time_range(l0, u0)
        assert rge.TimeInterval.json_loads(i0.json_dumps()) == i0
        assert rge.time_range(json.loads(i0.json_dumps())) == i0
        i = rge.EmptyTimeInterval()
        assert rge.TimeInterval.json_loads(i.json_dumps()) == i
        assert rge.time_range(json.loads(i.json_dumps())) == i
        assert rge.TimeInterval.json_loads('{"lower": "NaT"}') == i


class TestDiscreteRange(TestCase):

    def test_helper(self):
        item, items = 'item', ['i0', 'i1', 'i2']
        assert isinstance(rge.cat_range(item), rge.Level)
        assert isinstance(rge.cat_range(items), rge.Levels)

    def test_level(self):
        level = rge.Level('item')
        assert repr(level) == "<Level 'item'>"
        assert level == 'item'
        assert level.levels == ['item']

    def test_levels(self):
        levels = rge.Levels(['i0', 'i1', 'i2'])
        assert len(levels) == 3
        assert 'i0' in levels
        levels_repr = repr(levels._levels)
        assert repr(levels) == "<Levels {0}>".format(levels_repr)
        assert levels.levels == ['i0', 'i1', 'i2']

    def test_sql(self):
        levels = rge.Levels(['i0', 'i1'])
        sqls = ["x IN ('i0', 'i1')", "x IN ('i1', 'i0')"]
        assert levels.sql('x') in sqls
        level = rge.Level('item')
        sql = "x = 'item'"
        assert level.sql('x') == sql

    def test_intersection(self):
        levels = rge.cat_range(['i0', 'i1', 'i2'])
        others = rge.cat_range(['i0', 'i3'])
        empty = rge.Levels()
        l0 = rge.cat_range('i0')
        l3 = rge.cat_range('i3')
        assert levels & levels == levels
        assert levels & others == rge.cat_range(['i0'])
        assert l0 & levels == l0
        assert levels & l0 == l0
        assert l3 & l0 == empty
        assert l0 & l0 == l0
        assert empty & l0 == empty
        assert l0 & empty == empty

    def test_comparison(self):
        levels = rge.cat_range(['i0', 'i1', 'i2'])
        empty = rge.Levels()
        l0 = rge.cat_range('i0')
        l3 = rge.cat_range('i3')
        assert levels == levels
        assert levels != empty
        assert empty != levels
        assert l0 == l0
        assert l0 != l3
        assert l0 != empty
        assert l0 != levels
        assert levels != l0


def test_singletons():
    f = rge.float_range(3.0)
    assert isinstance(f, rge.FloatSingleton)
    assert repr(f) == "<FloatSingleton 3.0>"
    assert f.sql('x') == "x = 3.0"


class TestMultiFloatInterval(TestCase):

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
        assert rge.MultiFloatInterval.normalize(i0) == [i0]
        assert rge.MultiFloatInterval.normalize(i0, i1) == [r0]
        assert rge.MultiFloatInterval.normalize(i0, i1, i2) == [r0, i2]
        assert rge.MultiFloatInterval.normalize(i0, i1, i2, i3) == [r1]
        assert rge.MultiFloatInterval.normalize(i0, i1, i2, i3, i4) == [r2]
        assert rge.MultiFloatInterval.normalize(i5, i0, i1, i2, i3, i4) == [i5]

    def test_intersection(self):
        i0 = rge.FloatInterval(0, 1)
        i1 = rge.FloatInterval(0.5, 1.5)
        i2 = rge.FloatInterval(2, 3)
        i3 = rge.FloatInterval(0.25, 4)
        i4 = rge.FloatInterval(3.5, 5)
        r0 = rge.FloatInterval(0.5, 1)
        r1 = rge.FloatInterval(0.25, 1)
        r2 = rge.FloatInterval(3.5, 4)
        m1 = rge.MultiFloatInterval([i0, i4])
        m2 = rge.MultiFloatInterval([i2])
        assert rge.MultiFloatInterval.intersection(m1, m2) == \
            rge.MultiFloatInterval()
        m2 = rge.MultiFloatInterval([i1])
        assert rge.MultiFloatInterval.intersection(m1, m2) == \
            rge.MultiFloatInterval([r0])
        assert m1 & m2 == r0
        m2 = rge.MultiFloatInterval([i3])
        assert rge.MultiFloatInterval.intersection(m1, m2) == \
            rge.MultiFloatInterval([r1, r2])
        assert m1 & i3 == rge.MultiFloatInterval([r1, r2])
        assert i3 & m1 == rge.MultiFloatInterval([r1, r2])
        assert rge.FloatSingleton(1) & m1 == rge.EmptyFloatInterval()
        assert m1 & rge.FloatSingleton(1) == rge.EmptyFloatInterval()
        assert rge.FloatSingleton(0.25) & m1 == rge.FloatSingleton(0.25)
        assert m1 & rge.FloatSingleton(0.25) == rge.FloatSingleton(0.25)

    def test_difference(self):
        i0 = rge.FloatInterval(0, 1)
        i1 = rge.FloatInterval(0.5, 1.5)
        i2 = rge.FloatInterval(2, 3)
        i3 = rge.FloatInterval(0.25, 5)
        i4 = rge.FloatInterval(3.5, 5)
        r0 = rge.FloatInterval(0, 0.5)
        r1 = rge.FloatInterval(0, 0.25)
        m1 = rge.MultiFloatInterval([i0, i4])
        m2 = rge.MultiFloatInterval([i2])
        assert rge.MultiFloatInterval.difference(m1, m2) == m1
        m2 = rge.MultiFloatInterval([i1])
        assert rge.MultiFloatInterval.difference(m1, m2) == \
            rge.MultiFloatInterval([r0, i4])
        m2 = rge.MultiFloatInterval([i3])
        assert rge.MultiFloatInterval.difference(m1, m2) == \
            rge.MultiFloatInterval([r1])
        assert m1 - m2 == r1


if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
