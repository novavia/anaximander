#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxregistries.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import unittest
from unittest import TestCase

import anaximander as nx
from anaximander.utilities import functions as fun

#==============================================================================
### Mock NxObject
#==============================================================================


class Item(nx.Object):
    """ A dummy Object."""

    def __init__(self, ix=0):
        self.ix = ix

    def __repr__(self):
        return 'Item[{}]'.format(self.ix)

#==============================================================================
### Test Cases
#==============================================================================


class TestAttributeHandling(TestCase):

    def tests_lmap(self):
        i0, i1, i2 = (Item(i) for i in range(3))
        assert fun.lmap(i0, i2) == {'i0': i0, 'i2': i2}


class TestStringFormatting(TestCase):

    def test_spformat(self):
        c0 = []
        c1 = [Item()]
        c5 = [Item(i) for i in range(5)]
        assert fun.spformat(c0) == '0 item'
        assert fun.spformat(c1) == '1 item'
        assert fun.spformat(c5) == '5 items'
        assert fun.spformat(c1, 'thing', 'thangs') == '1 thing'
        assert fun.spformat(c5, 'thing') == '5 things'
        assert fun.spformat(len(c5), 'thing', 'thangs') == '5 thangs'

    def test_lformat(self):
        w = 4
        string = fun.lformat("A car has {w} wheels and {n} cylinders.")
        assert w == 4
        assert string == "A car has 4 wheels and {n} cylinders."


class TestMetaprogramming(TestCase):

    class Obj(nx.Object):
        pass

    class Mixin(object):

        def mixer(self):
            self.mixed = True

        @property
        def mixed(self):
            try:
                return self._mixed
            except AttributeError:
                return False

        @mixed.setter
        def mixed(self, val):
            self._mixed = bool(val)

        @classmethod
        def from_other(cls, obj):
            return cls()

    def test_ducktype(self):
        fun.ducktype(self.Obj, self.Mixin)
        m = self.Obj()
        m.mixer()
        n = self.Obj.from_other(m)
        assert m.mixed
        assert not n.mixed

if __name__ == '__main__':
    unittest.main()
