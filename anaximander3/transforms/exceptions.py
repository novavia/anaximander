"""
This module defines custom exceptions for Anaximander's transforms package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class TransformError(Exception):
    """A customized exception for transfomation errors."""
    pass


class InputError(TransformError):
    """Input-related exception."""
    pass
