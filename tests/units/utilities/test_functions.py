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

import pytest

from anaximander.utilities import functions as fun

# =============================================================================
# Mock Item
# =============================================================================


class Item(object):
    """ A dummy Object."""

    def __init__(self, ix=0):
        self.ix = ix

    def __repr__(self):
        return 'Item[{}]'.format(self.ix)

# =============================================================================
# Test Cases
# =============================================================================


class TestAttributeHandling(TestCase):

    def tests_lmap(self):
        i0, i1, i2 = (Item(i) for i in range(3))
        assert fun.lmap('i0', 'i2') == {'i0': i0, 'i2': i2}


def test_typecheck():
    assert fun.typecheck(int)(0)
    assert not fun.typecheck(int)('a')
    assert fun.typecheck(tuple, list)([])


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
        w, n = 4, '?'
        string = fun.lformat("My car has {w} wheels and {n} cylinders.")
        assert w == 4
        assert n == '?'
        assert string == "My car has 4 wheels and ? cylinders."

    def test_iformat(self):
        item = Item(0)
        assert fun.iformat('ix')(item) == "<Item ix:0>"
        assert fun.iformat()(item) == "<Item>"
        item.child = Item(1)
        assert fun.iformat('ix', 'child.ix')(item) == "<Item ix:0 child:1>"


class TestCollectionFunctions(TestCase):

    def test_dictunion(self):
        d0 = dict(a=1, b=2, c=3)
        d1 = dict(a=1, d=4)
        d2 = dict(e=5)
        union = fun.dictunion(d0, d1, d2)
        self.assertEqual(union, dict(a=1, b=2, c=3, d=4, e=5))
        with self.assertRaises(ValueError):
            union = fun.dictunion(d0, d1, unique_keys=True)


def test_pairwise():
    iterable = range(4)
    assert list(fun.pairwise(iterable)) == [(0, 1), (1, 2), (2, 3)]
    assert list(fun.pairwise(iterable, 2)) == [(0, 1), (2, 3)]
    assert list(fun.pairwise(iterable, 3)) == [(0, 1)]


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


@pytest.fixture
def klass():
    """Returns a generic class 'C'."""

    class C:
        c = 0

        def __repr__(self):
            return 'hello, world!'

    return C


def test_monkeypatch(klass):

    class Mixin:
        x = 0

    fun.monkeypatch(klass, Mixin)
    assert klass.__name__ == 'C'
    assert hasattr(klass, 'c')
    assert hasattr(klass, 'x')
    assert repr(klass()) == 'hello, world!'
    assert klass.__patches__[0] == Mixin


def test_monkeypatch_overwrite(klass):

    class Mixin:

        def __repr__(self):
            return 'no way, Jose!'

    # With exclusions
    fun.monkeypatch(klass, Mixin, '__repr__')
    assert repr(klass()) == 'hello, world!'
    # Without exclusions
    fun.monkeypatch(klass, Mixin)
    assert repr(klass()) == 'no way, Jose!'


if __name__ == '__main__':
    pytest.main([__file__])
