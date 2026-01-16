"""Anaximander AML package initialization."""

from .prototype import prototype, Archetype, Trait, Prototype
from .archetype import archetype
from .trait import trait
from .object import Object
from .namespaces import metadata, option, nxfield, meta


__all__ = [
    "prototype",
    "Archetype",
    "Trait",
    "Prototype",
    "archetype",
    "trait",
    "Object",
    "metadata",
    "option",
    "nxfield",
    "meta",
]
