#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxmeta.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import sys
import unittest
from unittest import TestCase

import anaximander as nx
from anaximander.meta import nxmeta
from anaximander.meta.nxmeta import typemethod
from anaximander.meta.basetypes import NxType, NxObject
from anaximander.utilities.functions import lmap

#==============================================================================
### Test Cases
#==============================================================================


class Testnxmeta(TestCase):

    def test_simple_programmatic_create(self):
        """Tests a simple type creation."""
        basecount = NxType.__typecount__
        C = NxType('C', (NxObject,), {})
        self.assertIs(type(C), NxType)
        self.assertTrue(issubclass(C, NxObject))
        self.assertIs(C.__module__, sys.modules[__name__])
        self.assertEqual(NxType.__typecount__, basecount + 1)

    def test_simple_declarative_create(self):
        """Tests declarative type creation."""
        class mockdescriptor(nxmeta.nxdescriptor):
            def __metainit__(self, cls):
                cls.mocks.append(self.name)

        class C(NxObject):
            mocks = []
            d1 = mockdescriptor()
            d2 = mockdescriptor()
            d3 = mockdescriptor()

        self.assertIs(type(C), NxType)
        self.assertEqual(C.mocks, ['d1', 'd2', 'd3'])

    def test_slots(self):
        """Tests class creation with slots argument."""
        C = NxType('C', (), {}, slots=('a', 'b'))
        inst = C()
        inst.a, inst.b = 0, 1
        with self.assertRaises(AttributeError):
            inst.x = 0
        self.assertEqual(inst.slots, (0, 1))
        inst.slots = (2, 3)
        self.assertEqual((inst.a, inst.b), (2, 3))


class Testtypemethod(TestCase):

    def test_getitem(self):
        """Tests a getitem typemethod."""
        class C(NxObject):
            @typemethod
            def __typegetitem__(cls, key):
                return key
        self.assertEqual(C[0], 0)

        class D(NxObject):
            @typemethod(target='__getitem__')
            def id_get(cls, key):
                return key
        self.assertEqual(D[0], 0)

        @typemethod
        def __typegetitem__(cls, key):
            return key
        patch = lmap('__typegetitem__')
        E = nxmeta.Assembler(NxObject, patch=patch)('E')
        self.assertEqual(E[0], 0)

        # Wrong function syntax
        with self.assertRaises(ValueError):
            class F(NxObject):
                @typemethod
                def typegetit(cls, key):
                    return key

        # Credible function syntax but no corresponding target
        with self.assertRaises(nxmeta.MetaError):
            class G(NxObject):
                @typemethod
                def __typegetit__(cls, key):
                    return key


class TestAssembler(TestCase):

    def test_create(self):
        """Tests a simple type creation."""
        a = nxmeta.Assembler(nx.nxobject)
        C = a('C')
        self.assertIs(type(C), nx.nxtype)
        self.assertIs(C.__module__, sys.modules[__name__])
        c = C()
        self.assertIs(type(c), C)
        self.assertIsInstance(c, nx.nxobject)

    def test_anonymous_create(self):
        """Tests type creation with programmatic name."""
        basecount = nx.Type.__typecount__
        a = nxmeta.Assembler(nx.Object)
        C = a()
        self.assertIs(type(C), nx.Type)
        self.assertIs(C.__module__, sys.modules[__name__])
        self.assertEqual(C.__name__, 'Type_' + str(basecount + 1))

if __name__ == '__main__':
    unittest.main(warnings='ignore')
