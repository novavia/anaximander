#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for data.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest

from anaximander.data import annotations as ant

# =============================================================================
# Test Cases
# =============================================================================


@pytest.fixture
def markertype():

    class MyMarker(ant.Marker):
        pass
    
    return MyMarker


def test_instantiation(markertype):
    red = markertype('red')
    assert list(markertype.shades.keys()) == ['red']
    assert markertype('red') == red


def test_declaration():

    class MyMarker(ant.Marker):
        red = ant.shade(color='red')
        green = ant.shade(color='green')
        
    assert list(MyMarker.shades.keys()) == ['red', 'green']
    assert MyMarker('red').plargs == {'color': 'red'}


if __name__ == '__main__':
    pytest.main([__file__])
