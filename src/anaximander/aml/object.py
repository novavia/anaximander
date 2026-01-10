"""This module defines the base object class for the Anaximander Modeling Language (AML)."""

from .prototype import Arche, prototype
from .archetype import archetype


@archetype
class Object(Arche, metaclass=prototype):
    """The base class for AML representation objects."""
    pass
