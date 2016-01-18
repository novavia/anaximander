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
        assert fun.lmap('i0', 'i2') == {'i0': i0, 'i2': i2}


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


class TestCollectionFunctions(TestCase):

    def test_dictunion(self):
        d0 = dict(a=1, b=2, c=3)
        d1 = dict(a=1, d=4)
        d2 = dict(e=5)
        union = fun.dictunion(d0, d1, d2)
        self.assertEqual(union, dict(a=1, b=2, c=3, d=4, e=5))
        with self.assertRaises(ValueError):
            union = fun.dictunion(d0, d1, unique_keys=True)


class TestDecorators(TestCase):

    def test_args_or_kwargs(self):

        @fun.args_or_kwargs
        def f(self, *args, **kwargs):
            if kwargs:
                return kwargs
            return dict(enumerate(args))

        @fun.args_or_kwargs(tabs=0)
        def g(x, y):
            return x + y

        with self.assertRaises(ValueError):
            f(Item(), 0, y=1)
        assert f(Item(), x=0, y=1) == {'x': 0, 'y': 1}
        assert f(Item(), 0, 1) == {0: 0, 1: 1}
        with self.assertRaises(ValueError):
            g(0, y=1)
        assert g(0, 1) == 1
        assert g(x=0, y=1) == 1


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

    def test_monkeypatch(self):
        fun.monkeypatch(self.Obj, self.Mixin)
        m = self.Obj()
        m.mixer()
        n = self.Obj.from_other(m)
        assert m.mixed
        assert not n.mixed

if __name__ == '__main__':
    unittest.main()
