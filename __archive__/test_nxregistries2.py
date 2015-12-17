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
from anaximander._meta import registries2 as reg

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


class TestNxBranchingNode(TestCase):

    def test_init(self):
        n = reg.NxBranchingNode()
        assert isinstance(n, reg.NxBranchingNode)

    def _build(self):
        """Builds an tree with items."""
        self.tree = reg.NxBranchingNode()
        i0, i1, i2, i3, i4, i5 = [Item(i) for i in range(6)]
        self.tree.register(i0, 'a')
        self.tree.register(i1, 'b', 'a')
        self.tree.register(i2, 'b', 'c')
        self.tree.register(i3, 'x', 'a', 'y')
        self.tree.register(i4, 'x', 'a', 'z')
        self.tree.register(i5, 'x', 'a')
        return i0, i1, i2, i3, i4, i5

    def test_insert(self):
        i0, i1, i2, i3, i4, i5 = self._build()
        assert self.tree['a'].value == i0
        assert isinstance(self.tree['b'], reg.NxBranchingNode)
        assert self.tree['b']['a'].value == i1

    def test_removal(self):
        i0, i1, i2, i3, i4, i5 = self._build()
        del self.tree['x']['a']['z']
        self.tree.discard('b')
        self.tree.unregister(i3, 'x', 'a', 'y')
        assert len(list(self.tree.values())) == 2
        assert not(set(self.tree.values()) ^ set([i0, i5]))

#    def test_copy(self):
#        i0, i1, i2, i3, i4 = self._build()
#        treecopy = self.tree.copy()
#        assert isinstance(treecopy['b'], reg.NxTree)
#        assert treecopy['b']['a'] == i1
#
#    def test_nodes(self):
#        i0, i1, i2, i3, i4 = self._build()
#        assert len(list(self.tree.nodes())) == 3
#        assert len(list(self.tree.nodes(1))) == 1
#        assert len(list(self.tree.nodes(2, 3, 1))) == 1
#        assert len(list(self.tree.nodes(2))) == 0
#
#    def test_fetch(self):
#        i0, i1, i2, i3, i4 = self._build()
#        with self.assertRaises(KeyError):
#            self.tree.fetch('e')
#        with self.assertRaises(ValueError):
#            self.tree.fetch('a')
#        assert self.tree.fetch('z') == i4
#        assert self.tree.fetch('x', 'y') == i3
#
#    def test_subset(self):
#        i0, i1, i2, i3, i4 = self._build()
#        assert bool(self.tree.subset('f')) is False
#        assert len(self.tree.subset('a').items()) == 4
#        assert len(self.tree.subset('c').items()) == 1
#        assert list(self.tree.subset('z').dictitems())[0] == ('z', i4)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
