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

    def test_insert(self):
        tree = nxregistries.NxTree()
        i0, i1 = Item(), Item()
        tree[0] = i0
        tree['a']['b'] = i1
        assert tree[0] == i0
        assert isinstance(tree['a'], nxregistries.NxTree)
        assert tree['a']['b'] == i1

    def test_copy(self):
        tree = nxregistries.NxTree()
        i0, i1, i2 = Item(), Item(), Item()
        tree[0] = i0
        tree[1][1] = i1
        tree[1][2] = i2
        treecopy = tree.copy()
        assert isinstance(treecopy[1], nxregistries.NxTree)
        assert treecopy[1][1] == i1

if __name__ == '__main__':
    unittest.main()