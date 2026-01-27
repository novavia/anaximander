"""This module defines the base object class for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================

from .archetype import archetype
from .interfaces import metadata
from .prototype import Arche, prototype

# =============================================================================
# Object archetype
# =============================================================================


@archetype
class Object(Arche, metaclass=prototype):
    """The base class for AML representation objects."""
    metadata.declare("pytype", type=type, nullable=False)
