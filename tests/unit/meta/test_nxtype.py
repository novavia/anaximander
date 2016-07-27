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

from anaximander.registries.folios import Registrable
from anaximander.meta.nxtype import NxType, nxtype

#==============================================================================
### Test Cases
#==============================================================================


class TestDeclarativeTypeCreation(TestCase):
    """Tests basic declarative metatype / type creation."""

    class ThingType(NxType, basename='Thing'):
        pass

    class Hammer(Registrable, metaclass=ThingType):
        pass

    class RedHammer(Hammer):
        pass

    def test_subclassing(self):
        """Tests subclassing NxType."""
        self.assertTrue(issubclass(self.ThingType, NxType))
        self.assertTrue(issubclass(self.ThingType, Registrable))

    def test_typing(self):
        """Tests creating a new type."""
        self.assertTrue(isinstance(self.Hammer, NxType))


class TestProgrammaticTypeCreation(TestCase):
    """Tests type creation with nxtype."""

    class ThingType(NxType, basename='Thing'):
        pass

    class BaseThing(Registrable, metaclass=ThingType):
        pass

    def test_typing(self):
        new_type = nxtype(self.BaseThing)
        self.assertTrue(issubclass(new_type, self.BaseThing))
        self.assertEqual(new_type.__name__, 'Thing_0')


if __name__ == '__main__':
    unittest.main(warnings='ignore')
