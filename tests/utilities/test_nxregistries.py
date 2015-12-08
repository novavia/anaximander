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
from anaximander.utilities import registries as reg

#==============================================================================
### Mock NxObject
#==============================================================================


class Item(nx.Object):
    """ A dummy Object."""

    def __init__(self, ix=0):
        self.ix = ix

#==============================================================================
### Test Cases
#==============================================================================


class TestNxCell(TestCase):

    def _build(self):
        """Builds an NxCell with items."""
        self.cell = reg.NxCell()
        i0, i1 = [Item() for i in range(2)]
        self.cell.register(i0)
        self.cell.register(i1)
        return i0, i1

    def test_insert(self):
        i0, i1 = self._build()
        assert len(self.cell) == 2

    def test_removal(self):
        i0, i1 = self._build()
        self.cell.unregister(i0)
        assert list(self.cell) == [i1]


class TestNxSortedCell(TestCase):

    def _build(self):
        """Builds an NxCell with items."""
        cell_type = reg.NxSortedCell[opr.attrgetter('ix')]
        self.cell = cell_type()
        i0, i1 = [Item(i) for i in range(2)]
        self.cell.register(i0)
        self.cell.register(i1)
        return i0, i1

    def test_insert(self):
        i0, i1 = self._build()
        assert len(self.cell) == 2

    def test_removal(self):
        i0, i1 = self._build()
        self.cell.unregister(i0)
        assert len(self.cell) == 1
        assert i1 in self.cell


class TestNxTree(TestCase):

    def _build(self):
        """Builds an NxTree with items."""
        self.tree = reg.NxTree()
        i0, i1, i2, i3, i4 = [Item() for i in range(5)]
        self.tree.register(i0, 'a')
        self.tree.register(i1, 'b', 'a')
        self.tree.register(i2, 'b', 'c')
        self.tree.register(i3, 'x', 'a', 'y')
        self.tree.register(i4, 'x', 'a', 'z')
        return i0, i1, i2, i3, i4

    def test_insert(self):
        i0, i1, i2, i3, i4 = self._build()
        assert self.tree['a'] == i0
        assert isinstance(self.tree['b'], reg.NxTree)
        assert self.tree['b']['a'] == i1

    def test_removal(self):
        i0, i1, i2, i3, i4 = self._build()
        del self.tree['x']['a']['z']
        self.tree.discard('b', 'a')
        self.tree.unregister(i2, 'b', 'c')
        assert len(self.tree.dictvalues()) == 2
        assert not(set(self.tree.values()) ^ set([i0, i3]))

    def test_copy(self):
        i0, i1, i2, i3, i4 = self._build()
        treecopy = self.tree.copy()
        assert isinstance(treecopy['b'], reg.NxTree)
        assert treecopy['b']['a'] == i1

    def test_nodes(self):
        i0, i1, i2, i3, i4 = self._build()
        assert len(list(self.tree.nodes())) == 3
        assert len(list(self.tree.nodes(1))) == 1
        assert len(list(self.tree.nodes(2, 3, 1))) == 1
        assert len(list(self.tree.nodes(2))) == 0

    def test_fetch(self):
        i0, i1, i2, i3, i4 = self._build()
        with self.assertRaises(KeyError):
            self.tree.fetch('e')
        with self.assertRaises(ValueError):
            self.tree.fetch('a')
        assert self.tree.fetch('z') == i4
        assert self.tree.fetch('x', 'y') == i3

    def test_subset(self):
        i0, i1, i2, i3, i4 = self._build()
        assert bool(self.tree.subset('f')) is False
        assert len(self.tree.subset('a').items()) == 4
        assert len(self.tree.subset('c').items()) == 1
        assert list(self.tree.subset('z').dictitems())[0] == ('z', i4)


class TestNxCellTree(TestCase):

    def _build(self):
        """Builds an NxTree with items."""
        cell_type = reg.NxSortedCell[opr.attrgetter('ix')]
        tree_type = reg.NxCellTree[cell_type]
        self.tree = tree_type()
        i0, i1, i2, i3, i4 = [Item() for i in range(5)]
        self.tree.register(i0, 'a')
        self.tree.register(i1, 'b')
        self.tree.register(i2, 'a')
        self.tree.register(i3, 'b')
        self.tree.register(i4, 'a')
        return i0, i1, i2, i3, i4

    def test_insert(self):
        i0, i1, i2, i3, i4 = self._build()
        assert i0 in self.tree['a']
        assert isinstance(self.tree['b'], reg.NxSortedCell)
        assert self.tree['b'][0] == i1

    def test_removal(self):
        i0, i1, i2, i3, i4 = self._build()
        del self.tree['a']
        self.tree.unregister(i1, 'b')
        assert len(self.tree.dictvalues()) == 1
        assert set(self.tree.values()) == set([i3])

    def test_copy(self):
        i0, i1, i2, i3, i4 = self._build()
        treecopy = self.tree.copy()
        assert isinstance(treecopy, reg.NxTree)
        assert treecopy['b'][0] == i1

    def test_fetch(self):
        i0, i1, i2, i3, i4 = self._build()
        with self.assertRaises(KeyError):
            self.tree.fetch('e')
        assert list(self.tree.fetch('a')) == [i0, i2, i4]

if __name__ == '__main__':
    unittest.main(warnings='ignore')
