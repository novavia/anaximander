"""Archetypes and traits for AML data types."""

# =============================================================================
# Imports
# =============================================================================

from typing import TYPE_CHECKING, ClassVar

from .archetype import archetype
from .interfaces import metadata
from .metadescriptors import Metadescriptor
from .object import Object
from .protodescriptors import ParserDeclarator, ValidatorDeclarator
from .trait import trait

# =============================================================================
# Data archetypes
# =============================================================================


@archetype
class Data(Object):
    """Base archetype for scalar data types."""

    __declarator_types__ = {Metadescriptor, ParserDeclarator, ValidatorDeclarator}


@archetype
class Scalar[T](Data):
    """Base archetype for scalar data types with concrete Python materialization."""
    pass


class Integer(Scalar[int], int):
    """Integer scalar archetype."""
    pass


class Float(Scalar[float], float):
    """Float scalar archetype."""
    pass


class Bool(Scalar[bool]):
    """Boolean scalar archetype."""
    pass

if TYPE_CHECKING:
    Bool = bool  # type: ignore[assignment]


class String(Scalar[str], str):
    """String scalar archetype."""
    pass


# =============================================================================
# Measurement trait and archetype
# =============================================================================


@trait
class measurement(Data):
    """Trait for measurements that declare a physical unit."""
    unit: ClassVar[str] = metadata()


@archetype
class Measurement(Scalar[float], traits=(measurement,)):
    """Measurement archetype with unit metadata and float materialization."""
    pass
