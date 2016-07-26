#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for nxtype.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import unittest
from unittest import TestCase

from anaximander.meta import nxobject
from anaximander.meta.nxtype import NxType, nxtype

#==============================================================================
### Test Cases
#==============================================================================


class TestDeclarativeTypeCreation(TestCase):
    """Tests basic declarative metatype / type creation."""

    class ThingType(NxType, basename='Thing'):
        pass

    class Hammer(nxobject, metaclass=ThingType):
        pass

    class RedHammer(Hammer):
        pass

    def test_subclassing(self):
        """Tests subclassing NxType."""
        self.assertTrue(issubclass(self.ThingType, NxType))
        self.assertTrue(issubclass(self.ThingType, nxobject))

    def test_typing(self):
        """Tests creating a new type."""
        self.assertTrue(isinstance(self.Hammer, NxType))

    def test_folios_property(self):
        """Verifies that the folios property works as intended.
        
        In particular, tests the independence of the property between
        a type (Hammer) and its instances.
        """
        h = self.Hammer()
        rh = self.RedHammer()
        h._folios.add(0)
        rh._folios.add(1)
        self.RedHammer._folios.add(2)
        self.assertEqual(h._folios, {0})
        self.assertEqual(self.Hammer._folios, set())
        self.assertEqual(rh._folios, {1})
        self.assertEqual(self.RedHammer._folios, {2})


class TestProgrammaticTypeCreation(TestCase):
    """Tests type creation with nxtype."""

    class ThingType(NxType, basename='Thing'):
        pass

    class BaseThing(nxobject, metaclass=ThingType):
        pass

    def test_typing(self):
        new_type = nxtype(self.BaseThing)
        self.assertTrue(issubclass(new_type, self.BaseThing))
        self.assertEqual(new_type.__name__, 'Thing_0')


if __name__ == '__main__':
    unittest.main(warnings='ignore')
