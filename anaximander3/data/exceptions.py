"""
This module defines custom exceptions for Anaximander's data package.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""


class SchemaError(Exception):
    """A customized exception for Schema construction errors."""
    pass


class ConformityError(Exception):
    """Raised if data supplied to a DataObject doesn't conform."""
    pass
