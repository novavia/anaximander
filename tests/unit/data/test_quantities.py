#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for quantities.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander.data.quantities import Quantity

# =============================================================================
# Test Cases
# =============================================================================


def test_instantiation():
    speed = Quantity('speed', 'mph')
    assert Quantity['speed'].unit == 'mph'


if __name__ == '__main__':
    pytest.main([__file__])
