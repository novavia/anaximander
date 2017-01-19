#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom exception classes for the data package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class DataError(Exception):
    """Base exception class for the data package."""
    pass


class ValidationError(DataError):
    """Raised when data fails a validation test."""
    pass
