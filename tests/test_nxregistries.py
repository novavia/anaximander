#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxregistries.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
# Import statements
#==============================================================================

import unittest
from unittest import TestCase

import nxregistries

#==============================================================================
# Mock NxObject
#==============================================================================

class Item(object):
    """ A dummy object that mocks as an NxObject."""

    def __init__(self):
        self._nxregistries = set()

#==============================================================================
# NxTree testing
#==============================================================================

class TestNxTree(TestCase):

    def _build(self):
        """Builds an NxTree with items."""
        self.tree = nxregistries.NxTree()
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
        assert isinstance(self.tree['b'], nxregistries.NxTree)
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
        assert isinstance(treecopy['b'], nxregistries.NxTree)
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

if __name__ == '__main__':
    unittest.main()