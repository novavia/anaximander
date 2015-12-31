#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for registries.

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
    """A registry with mixed laypeyers for testing purposes."""
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
        with self.assertRaises(ValueError):  # Improper use of args / kwargs
            registry.register(i2, 'a', 2, key=2)
        registry.register(i2, 'a', 2, 2)
        assert i2._folios == {registry['a'][2]}
        registry.register(i3, 'a')
        assert registry['a'].title == i3
        with self.assertRaises(KeyError):
            registry.register(i4, 'a', 1)  # Missing key.
        with self.assertRaises(KeyError):
            registry.register(i4, 'a', 'b', 1)  # Wrong key type 'b'.
        with self.assertRaises(KeyError):
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
        i0, i1, i2, i3 = (Item(i) for i in range(4))
        registry.register(i0, 'a', 1, 0)
        registry.register(i1, 'a', 1, 1)
        registry.register(i2, 'a', 2, 2)
        registry.register(i3, 'b', 2, 0)
        assert set(registry.branches()) == {('a', 1), ('a', 2), ('b', 2)}
        assert set(registry.branches('b')) == {('b', 2)}
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
        assert s2.hardcopy() == {'a': {2: {2: i2}},
                                 'b': {2: {0: i3}}}
        assert s3.root.subcount == 5
        assert not s4.root
        ks0 = registry.subset(schedule=2)
        ks1 = registry.subset(volume='a')
        ks2 = registry.subset(volume='x')
        assert ks0.hardcopy() == {'a': {2: {2: i2}},
                                  'b': {2: {0: i3}}}
        assert ks1.root.subcount == 3
        assert ks2.root.subcount == 0

    def test_getters(self):
        registry = ComplexRegistry()
        i0, i1, i2, i3 = (Item(i) for i in range(4))
        registry.register(i0, 'a', 1, 0)
        registry.register(i1, 'a', 1, 1)
        registry.register(i2, 'a', 2, 2)
        registry.register(i3, 'b', 2, 0)
        assert set(registry.addresses()) == {('a', 1, 0), ('a', 1, 1),
                                             ('a', 2, 2), ('b', 2, 0)}
        assert set(registry.addresses(volume='b')) == {('b', 2, 0)}
        with self.assertRaises(KeyError):
            registry.addresses('b', 2, 0)
        assert set(registry.values()) == {i0, i1, i2, i3}
        assert set(registry.values('b')) == {i3}
        assert set(registry.items('b')) == {(('b', 2, 0), i3)}
        assert set(registry.scan(2)) == {i2, i3}


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
        assert set(registry.branches('a', 'c')) == {('a', 'c')}
        assert registry.unregister(i1, 'a', 'c')
        assert registry.root.height == 2
        assert registry.root.subcount == 2

    def test_branch_and_subset(self):
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

    def test_getters(self):
        registry = RecursiveRegistry()
        i0, i1, i2 = (Item(i) for i in range(3))
        registry.register(i0, 'a', 'b')
        registry.register(i1, 'a', 'c')
        registry.register(i2, 'b', 'a')
        assert set(registry.values()) == set()
        assert set(registry.titles()) == {i0, i1, i2}
        assert set(registry.titles('a')) == {i0, i1}
        assert set(registry.titles('a', 'b')) == {i0}
        assert set(registry.browse('b')) == {i0, i2}
        with self.assertRaises(ValueError):
            registry.get('a')
        assert registry.get('a', 'b') == i0
        with self.assertRaises(KeyError):
            registry.get('c')
        with self.assertRaises(ValueError):
            registry.fetch('b')
        assert registry.fetch('c') == i1
        with self.assertRaises(KeyError):
            registry.fetch('d')

if __name__ == '__main__':
    unittest.main(warnings='ignore')
