"""This module defines the base object class for the Anaximander Modeling Language (AML)."""

from .meta import Arche, Type
from .archetype import archetype


@archetype
class Object[T](Arche, metaclass=Type):
    """The base class for AML representation objects."""
    pass
