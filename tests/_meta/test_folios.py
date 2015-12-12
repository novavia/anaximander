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

import operator as opr
import unittest
from unittest import TestCase

import anaximander as nx
from anaximander._meta import folios

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


class TestNxPortFolio(TestCase):

    def test_construct(self):
        """Tests basic building blocks."""
        root = folios.NxFolder()
        a, ab, cde = (folios.NxFolder() for i in range(3))
        root['a'] = a
        root['a']['b'] = ab
        root.insert(cde, 'c', 'd', 'e')
        root['f'] = folios.NxFolder()  # Value doesn't hold
        assert root.height == 3
        assert root.subcount == 5
        item = Item(0)
        with self.assertRaises(TypeError):
            root['f'] = item
        root['f'] = folios.NxFolder(item)
        assert root.subcount == 6

    def test_titles(self):
        """Tests titled folders."""
        root = folios.NxFolder()
        i0, i1, i2 = (Item(i) for i in range(3))
        root['a'] = folios.NxFolder(i0)
        root['b'] = folios.NxFolder(title=i1)
        root['b'].sub('c', title=i2)
        assert root.subcount == 3
        assert i2._folios == set([root['b']['c']])
        root['b'].dispose()
        assert not i2._folios
        assert list(root.keys()) == ['a']
        root['a'].title = i1
        assert not i0._folios
        assert i1._folios == set([root['a']])

    def test_find(self):
        """Tests find and fetch."""
        root = folios.NxFolder()
        a, ab, cde = (folios.NxFolder() for i in range(3))
        root['a'] = a
        root['a']['b'] = ab
        root.insert(cde, 'c', 'd', 'a')
        afolios = list(root.find('a'))
        assert afolios == [root['a'], root.get('c', 'd', 'a')]
        with self.assertRaises(ValueError):
            root.fetch('a')
        with self.assertRaises(KeyError):
            root.fetch('x')
        assert root.fetch('d') == root['c']['d']


class TestVolume(TestCase):

    def test_numbering(self):
        """Tests page numbering."""
        root = folios.NxVolume()
        p0, p1, p2 = (folios.NxPage() for i in range(3))
        root.insert(p0)
        root.insert(p1, 3)
        root.insert(p2)
        assert root[1] == p0
        assert root[3] == p1
        assert root[4] == p2


class TestCard(TestCase):

    def test_entry(self):
        """Tests input and disposal."""
        root = folios.NxFolder()
        i = Item()
        card = folios.NxCard(i)
        root['a'] = card
        assert i._folios == set([card])
        root.dispose()
        assert i._folios == set()


class TestPage(TestCase):

    def test_entries(self):
        """Tests input and disposal."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxPage([i0, i1, i2])
        assert i0._folios == set([p])
        p.remove(i0)
        assert i0._folios == set()
        with self.assertRaises(KeyError):
            p.remove(i0)
        p.discard(i0)
        p.dispose()
        assert i1._folios == set()

    def test_iterate(self):
        """Tests iterating on entries."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxPage([i0, i1, i2])
        entries = p.entries()
        p.dispose()
        assert set(entries) == set([i0, i1, i2])


class TestScroll(TestCase):

    def test_entries(self):
        """Tests input and disposal."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxScroll([i0, i1, i2], key=opr.attrgetter('ix'))
        assert i0._folios == set([p])
        p.remove(i0)
        assert i0._folios == set()
        with self.assertRaises(KeyError):
            p.remove(i0)
        p.discard(i0)
        p.dispose()
        assert i1._folios == set()

    def test_iterate(self):
        """Tests iterating on entries."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxScroll([i0, i1, i2], key=opr.attrgetter('ix'))
        entries = p.entries()
        p.dispose()
        assert list(entries) == [i0, i1, i2]

if __name__ == '__main__':
    unittest.main(warnings='ignore')
