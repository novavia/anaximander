#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for schema.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander2.fields import Integer, Text
from anaximander2.columnar.schema import FieldMap

# =============================================================================
# Tests
# =============================================================================


class MyFieldMap(FieldMap):
    x = Integer()
    y = Text()


def test_field_map():
    assert isinstance(MyFieldMap['x'], Integer)
    assert isinstance(MyFieldMap['y'], Text)
    f = MyFieldMap()
    assert isinstance(f['x'], Integer)
    assert isinstance(f['y'], Text)

if __name__ == '__main__':
    pytest.main([__file__, '-x', '--pdb'])
