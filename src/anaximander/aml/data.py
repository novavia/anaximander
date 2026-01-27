"""Archetypes and traits for AML data types."""

# =============================================================================
# Imports
# =============================================================================

from typing import Any, ClassVar

from .archetype import archetype
from .declarative import is_not_missing
from .interfaces import metadata
from .metadescriptors import Metadescriptor
from .object import Object
from .protodescriptors import ParserDeclarator, ValidatorDeclarator
from .prototype import prototype
from .trait import trait

# =============================================================================
# Data archetypes
# =============================================================================


@archetype
class Data(Object):
    """Base archetype for scalar data types."""

    __declarator_types__ = {Metadescriptor, ParserDeclarator, ValidatorDeclarator}

    @classmethod
    def value_parsers(cls) -> tuple[ParserDeclarator, ...]:
        """Return parsers that apply to the data value itself."""
        constructors = cls.__merged_metacharacters__.constructor.values()
        parsers = [
            p for p in constructors if isinstance(p, ParserDeclarator) and not p.members
        ]
        return tuple(parsers)

    @classmethod
    def value_validators(cls) -> tuple[ValidatorDeclarator, ...]:
        """Return validators that apply to the data value itself."""
        constructors = cls.__merged_metacharacters__.constructor.values()
        validators = [
            v for v in constructors if isinstance(v, ValidatorDeclarator) and not v.members
        ]
        return tuple(validators)

    @classmethod
    def parse_value(cls, value: Any) -> Any:
        """Apply value parsers to a data value."""
        parsed = value
        for parser in cls.value_parsers():
            if is_not_missing(parser.callable):
                parsed = parser.callable(cls, parsed)
        return parsed

    @classmethod
    def validate_value(cls, value: Any) -> bool:
        """Validate a data value with value validators."""
        for validator in cls.value_validators():
            if is_not_missing(validator.callable) and not validator.callable(cls, value):
                return False
        return True


@archetype
class Scalar(Data):
    """Base archetype for scalar data types with concrete Python materialization."""
    pass


@archetype
class Integer(Scalar):
    """Integer scalar archetype."""
    pytype = int


@archetype
class Float(Scalar):
    """Float scalar archetype."""
    pytype = float


@archetype
class Bool(Scalar):
    """Boolean scalar archetype."""
    pytype = bool


@archetype
class String(Scalar, metaclass=prototype):
    """String scalar archetype."""
    pytype = str


# =============================================================================
# Measurement trait and archetype
# =============================================================================


@trait
class MeasurementTrait(Data, metaclass=prototype):
    """Trait for measurements that declare a physical unit."""
    unit: ClassVar[str] = metadata()


@archetype
class Measurement(Float, metaclass=prototype, traits=(MeasurementTrait,)):
    """Measurement archetype with unit metadata and float materialization."""
    pass
