"""Anaximander AML package initialization."""

from .meta import Type, Archetype, Trait, Prototype
from .archetype import archetype
from .trait import trait
from .object import Object


__all__ = [
    "Type",
    "Archetype",
    "Trait",
    "Prototype",
    "archetype",
    "trait",
    "Object",
]
