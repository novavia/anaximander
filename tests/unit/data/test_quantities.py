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

from anaximander.data import quantities as qnt

# =============================================================================
# Test Cases
# =============================================================================


def test_instantiation():
    qnt.Quantity('speed', 'mph')
    assert qnt.Quantity['speed'].unit == 'mph'
    assert qnt.speed == qnt.Quantity['speed']
    qnt.Quantity('speed', 'kph')
    assert qnt.Quantity['speed'].unit == 'kph'
    assert qnt.speed == qnt.Quantity['speed']
    with pytest.raises(TypeError):
        qnt.Quantity(0, 0)

if __name__ == '__main__':
    pytest.main([__file__])
