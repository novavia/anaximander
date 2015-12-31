#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for xtypes.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

#==============================================================================
### Imports
#==============================================================================

import unittest
from unittest import TestCase

import anaximander as nx
from anaximander._meta import xtypes

#==============================================================================
### Mock NxObject
#==============================================================================


class Item(nx.Object):
    pass

#==============================================================================
### Tests
#==============================================================================


class Test_metatype(TestCase):
    """Tests the extended type metatype."""

    class MetaType(nx.Type, xtypes.metatype):
        pass

    class MetaItem(Item, metaclass=MetaType):
        pass

    def test_subtype(self):
        cls = self.MetaItem.subtype()
        assert type(cls) == self.MetaType
        assert cls.__name__ == 'MetaItem'

if __name__ == '__main__':
    unittest.main(warnings='ignore')
