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
from anaximander._meta import folios as fol, registries as nrg

#==============================================================================
### Mock NxObject
#==============================================================================


class Item(nx.Object):
    """ A dummy Object."""

    def __init__(self, ix=0):
        self.ix = ix

    def __repr__(self):
        return 'Item[{}]'.format(self.ix)


class ComplexRegistry(nrg.NxRegistry):
    """A registry with mixed layers for testing purposes."""
    __root__ = fol.NxFolder
    __layers__ = [('volume', fol.NxVolume),
                  ('schedule', fol.NxSchedule)]


class RecursiveRegistry(nrg.NxRegistry):
    """A recursive layer registry for testing purposes."""
    __root__ = fol.NxFolder
    __recurse__ = fol.NxFolder

#==============================================================================
### Test Cases
#==============================================================================


class TestComplexRegistry(TestCase):

    def test_register(self):
        """Tests basic registration / unregistration."""
        registry = ComplexRegistry()
        i0, i1, i2, i3, i4 = (Item(i) for i in range(5))
        registry.register(i0, 'a', 1, 0)
        assert i0 in list(registry['a'][1].read())
        registry.register(i1, volume='a', schedule=1, key=1)
        assert len(registry['a'][1]) == 2
        registry.register(i2, 'a', 2, key=2)
        assert i2._folios == {registry['a'][2]}
        registry.register(i3, 'a')
        assert registry['a'].title == i3
        with self.assertRaises(ValueError):
            registry.register(i4, 'a', 1)  # Missing key.
        with self.assertRaises(ValueError):
            registry.register(i4, 'a', 'b', 1)  # Wrong key type 'b'.
        with self.assertRaises(ValueError):
            registry.register(i3, vol='a', schedule=1, key=3)  # Wrong names.
        assert not registry.unregister(i0)
        assert len(registry['a'][1]) == 2
        assert registry.unregister(i0, 'a', 1, 0)
        assert len(registry['a'][1]) == 1
        assert registry.unregister(i2, 'a', 2, 2)
        assert len(registry['a']) == 1
        assert not i0._folios
        assert not i2._folios

    def test_copy_and_delete(self):
        registry = ComplexRegistry()
        i0 = Item(0)
        registry.register(i0, 'a', 1, 0)
        copy = registry.copy()
        assert i0 in list(copy['a'][1].read())
        assert len(i0._folios) == 2
        del registry
        assert len(i0._folios) == 1

    def test_branch_and_subset(self):
        registry = ComplexRegistry()
        i0, i1, i2 = (Item(i) for i in range(3))
        registry.register(i0, 'a', 1, 0)
        registry.register(i1, 'a', 1, 1)
        registry.register(i2, 'a', 2, 2)
        b0 = registry.branch('a')
        b1 = registry.branch(volume='a', schedule=1)
        assert b0['a'][1].hardcopy() == {0: i0, 1: i1}
        assert len(i0._folios) == 1
        assert b0.root.height == 2
        assert b0.root.subcount == 3
        assert b1.root.height == 2
        assert b1.root.subcount == 2
        s0 = registry.subset('a')
        s1 = registry.subset(1)
        s2 = registry.subset(2)
        s3 = registry.subset()
        s4 = registry.subset('x')
        assert s0.root.subcount == 3
        assert len(s1['a']) == 1
        assert s2.hardcopy() == {'a': {2: {2: i2}}}
        assert s3.root.subcount == 3
        assert not s4.root


class TestRecursiveRegistry(TestCase):

    def test_register(self):
        """Tests basic registration / unregistration."""
        registry = RecursiveRegistry()
        i0, i1, i2 = (Item(i) for i in range(3))
        registry.register(i0, 'a', 'b')
        assert registry['a']['b'].title == i0
        registry.register(i1, 'a', 'c')
        assert registry['a']['c'].title == i1
        registry.register(i2, 'a', 'b')
        assert registry['a']['b'].title == i2
        assert not i0._folios
        assert i2._folios == {registry['a']['b']}
        assert registry.unregister(i1, 'a', 'c')
        assert registry.root.height == 2
        assert registry.root.subcount == 2

    def test_branch(self):
        registry = RecursiveRegistry()
        i0, i1, i2 = (Item(i) for i in range(3))
        registry.register(i0, 'a', 'b')
        registry.register(i1, 'a', 'c')
        registry.register(i2, 'b', 'a')
        b0 = registry.branch('a')
        b1 = registry.branch('a', 'b')
        b2 = registry.branch('b')
        assert b0.root.subcount == 3
        assert b1.root.subcount == 2
        assert b2.root.subcount == 2

if __name__ == '__main__':
    unittest.main(warnings='ignore')
