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


class TestNxFolder(TestCase):

    def test_construct(self):
        """Tests basic building blocks."""
        root = folios.NxFolder()
        a, ab, cde = (folios.NxFolder() for i in range(3))
        root['a'] = a
        root['a']['b'] = ab
        root.insert(cde, 'c', 'd', 'e')
        root['f'] = folios.NxFolder()  # Value doesn't hold
        root_copy = root.copy()
        assert root.height == root_copy.height == 3
        assert root.subcount == root_copy.subcount == 5
        assert set(root.subfolios(0)) == {root}
        assert set(root.subfolios(1)) == {root['a'], root['c']}
        assert set(root.subfolios(0, 2, 3)) == {root,
                                                root['a']['b'],
                                                root['c']['d'],
                                                root['c']['d']['e']}
        assert root.path == ()
        assert root.tab is None
        assert root['c']['d']['e'].path == ('c', 'd', 'e')
        assert root['c']['d']['e'].tab == 'e'
        item = Item(0)
        with self.assertRaises(TypeError):
            root['f'] = item
        root['f'] = folios.NxFolder(item)
        assert root.subcount == 6
        assert root_copy.subcount == 5

    def test_titles(self):
        """Tests titled folders."""
        root = folios.NxFolder()
        i0, i1, i2 = (Item(i) for i in range(3))
        root['a'] = folios.NxFolder(i0)
        root['b'] = folios.NxFolder(title=i1)
        root['b'].sub('c', title=i2)
        root_copy = root.copy()
        assert root.subcount == 3
        assert i2._folios == {root['b']['c']}
        assert root_copy['b']['c'].title == i2
        assert set(root.titles()) == {i0, i1, i2}
        assert set(root.titles(2, root=True)) == {i2}
        root['b'].dispose()
        assert not i2._folios
        assert list(root.keys()) == ['a']
        assert list(root_copy.keys()) == ['a']
        root['a'].title = i1
        assert not i0._folios
        assert i1._folios == {root['a']}
        assert root_copy['a'].title == i1

    def test_find(self):
        """Tests search and find."""
        root = folios.NxFolder()
        a, ab, cde = (folios.NxFolder() for i in range(3))
        root['a'] = a
        root['a']['b'] = ab
        root.insert(cde, 'c', 'd', 'a')
        a_copy = root['a'].copy()
        afolios = list(root.search('a'))
        assert afolios == [root['a'], root.retrieve('c', 'd', 'a')]
        bfolios = list(a_copy.search('b'))
        assert bfolios == [a_copy['b']]
        with self.assertRaises(ValueError):
            root.find('a')
        with self.assertRaises(KeyError):
            root.find('x')
        assert root.find('d') == root['c']['d']
        assert a_copy.find('b') == a_copy['b']

    def test_copy(self):
        """Tests all copy forms."""
        root = folios.NxFolder()
        i0, i1, i2 = (Item(i) for i in range(3))
        root['a'] = folios.NxFolder(i0)
        root['b'] = folios.NxFolder(title=i1)
        root['b'].sub('c', title=i2)
        copy = root.copy()
        deepcopy = root.deepcopy()
        hardcopy = root.hardcopy()
        copy_of_copy = copy.copy()
        deepcopy_of_copy = copy.deepcopy()
        hardcopy_of_copy = copy.hardcopy()
        copy_of_deepcopy = deepcopy.copy()
        assert copy['b']['c'].title == i2
        assert root._proxies == {copy, copy_of_copy}
        assert copy_of_copy['b']['c'].title == i2
        assert copy_of_deepcopy['b']['c'].title == i2
        assert deepcopy._proxies == {copy_of_deepcopy}
        assert i0._folios == {root['a'], deepcopy['a'], deepcopy_of_copy['a']}
        assert hardcopy['b']['c'].title == i2
        assert hardcopy_of_copy['b']['c'].title == i2


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
        assert root[1].path == (1,)


class TestCard(TestCase):

    def test_entry(self):
        """Tests input and disposal."""
        root = folios.NxFolder()
        i = Item()
        card = folios.NxCard(i)
        root['a'] = card
        assert card.path == ('a',)
        assert i._folios == {card}
        root.dispose()
        assert i._folios == set()

    def test_copy(self):
        """Tests different copy types."""
        root = folios.NxFolder()
        i = Item()
        card = folios.NxCard(i)
        root['a'] = card
        copy = root.copy()
        deepcopy = root.deepcopy()
        hardcopy = root.hardcopy()
        copy_of_copy = copy.copy()
        deepcopy_of_copy = copy.deepcopy()
        hardcopy_of_copy = copy.hardcopy()
        copy_of_deepcopy = deepcopy.copy()
        assert copy['a'].entry == i
        assert copy_of_copy['a'].entry == i
        assert copy_of_deepcopy['a'].entry == i
        assert root._proxies == {copy, copy_of_copy}
        assert deepcopy._proxies == {copy_of_deepcopy}
        assert i._folios == {card, deepcopy['a'], deepcopy_of_copy['a']}
        assert hardcopy == {'a': i}
        assert hardcopy_of_copy == {'a': i}


class TestPage(TestCase):

    def test_entries(self):
        """Tests input and disposal."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxPage([i0, i1, i2])
        assert i0._folios == {p}
        p.remove(i0)
        assert i0._folios == set()
        with self.assertRaises(KeyError):
            p.remove(i0)
        p.discard(i0)
        del i2
        assert len(p) == 1
        p.dispose()
        assert i1._folios == set()

    def test_iterate(self):
        """Tests iterating on entries."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxPage([i0, i1, i2])
        entries = p.read()
        p.dispose()
        assert set(entries) == set([i0, i1, i2])

    def test_copy(self):
        """Tests different copy types."""
        i0, i1, i2 = (Item(i) for i in range(3))
        p = folios.NxPage([i0, i1, i2])
        copy = p.copy()
        deepcopy = p.deepcopy()
        hardcopy = p.hardcopy()
        copy_of_copy = copy.copy()
        deepcopy_of_copy = copy.deepcopy()
        hardcopy_of_copy = copy.hardcopy()
        copy_of_deepcopy = deepcopy.copy()
        assert set(copy.read()) == {i0, i1, i2}
        assert set(copy_of_copy.read()) == {i0, i1, i2}
        assert set(copy_of_deepcopy.read()) == {i0, i1, i2}
        assert p._proxies == {copy, copy_of_copy}
        assert deepcopy._proxies == {copy_of_deepcopy}
        assert i0._folios == {p, deepcopy, deepcopy_of_copy}
        assert hardcopy == {i0, i1, i2}
        assert hardcopy_of_copy == {i0, i1, i2}


class TestScroll(TestCase):

    def test_entries(self):
        """Tests input and disposal."""
        i0, i1, i2 = (Item(i) for i in range(3))
        s = folios.NxScroll([i0, i1, i2], key=opr.attrgetter('ix'))
        assert i0._folios == {s}
        s.remove(i0)
        assert i0._folios == set()
        with self.assertRaises(KeyError):
            s.remove(i0)
        s.discard(i0)
        del i2
        # This is used in lieu of len(s) because the weaksorted set
        # doesn't catch on immediately to the disappearance of i2.
        assert len(list(s.read())) == 1
        s.dispose()
        assert i1._folios == set()

    def test_iterate(self):
        """Tests iterating on entries."""
        i0, i1, i2 = (Item(i) for i in range(3))
        s = folios.NxScroll([i0, i1, i2], key=opr.attrgetter('ix'))
        entries = s.read()
        s.dispose()
        assert list(entries) == [i0, i1, i2]

    def test_copy(self):
        """Tests different copy types."""
        i0, i1, i2 = (Item(i) for i in range(3))
        s = folios.NxScroll([i0, i1, i2], key=opr.attrgetter('ix'))
        copy = s.copy()
        deepcopy = s.deepcopy()
        hardcopy = s.hardcopy()
        copy_of_copy = copy.copy()
        deepcopy_of_copy = copy.deepcopy()
        hardcopy_of_copy = copy.hardcopy()
        copy_of_deepcopy = deepcopy.copy()
        assert set(copy.read()) == {i0, i1, i2}
        assert set(copy_of_copy.read()) == {i0, i1, i2}
        assert set(copy_of_deepcopy.read()) == {i0, i1, i2}
        assert s._proxies == {copy, copy_of_copy}
        assert deepcopy._proxies == {copy_of_deepcopy}
        assert i0._folios == {s, deepcopy, deepcopy_of_copy}
        assert set(hardcopy) == {i0, i1, i2}
        assert set(hardcopy_of_copy) == {i0, i1, i2}


class TestSchedule(TestCase):

    def test_entries(self):
        """Tests input and disposal."""
        i0, i1, i2 = (Item(i) for i in range(3))
        d = folios.NxSchedule(enumerate([i0, i1, i2]))
        assert d.path == ()
        assert i0._folios == {d}
        del d[0]
        assert i0._folios == set()
        with self.assertRaises(KeyError):
            del d[0]
        del i2
        assert len(d) == 1
        d.dispose()
        assert i1._folios == set()

    def test_iterate(self):
        """Tests iterating on entries."""
        i0, i1, i2 = (Item(i) for i in range(3))
        d = folios.NxSchedule(enumerate([i0, i1, i2]))
        entries = d.read()
        d.dispose()
        assert list(entries) == [i0, i1, i2]

    def test_copy(self):
        """Tests different copy types."""
        i0, i1, i2 = (Item(i) for i in range(3))
        d = folios.NxSchedule(enumerate([i0, i1, i2]))
        copy = d.copy()
        deepcopy = d.deepcopy()
        hardcopy = d.hardcopy()
        copy_of_copy = copy.copy()
        deepcopy_of_copy = copy.deepcopy()
        hardcopy_of_copy = copy.hardcopy()
        copy_of_deepcopy = deepcopy.copy()
        assert set(copy.read()) == {i0, i1, i2}
        assert set(copy_of_copy.read()) == {i0, i1, i2}
        assert set(copy_of_deepcopy.read()) == {i0, i1, i2}
        assert d._proxies == {copy, copy_of_copy}
        assert deepcopy._proxies == {copy_of_deepcopy}
        assert i0._folios == {d, deepcopy, deepcopy_of_copy}
        assert hardcopy == {0: i0, 1: i1, 2: i2}
        assert hardcopy_of_copy == {0: i0, 1: i1, 2: i2}

if __name__ == '__main__':
    unittest.main(warnings='ignore')
