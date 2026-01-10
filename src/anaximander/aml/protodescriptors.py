"""This module defines the Protodescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import datetime
import re
from abc import abstractmethod
from collections.abc import Mapping
from numbers import Real
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Literal,
)

import yaml
from attrs import field

from .declarative import Declarator, DeclaratorRegistry, declarator

# endregion

# =============================================================================
# Constants and Utilities
# =============================================================================
# region Constants and Utilities


# Sentinel value for unspecified defaults
class _MissingSentinel:
    """Unique sentinel for unspecified defaults."""
    def __repr__(self) -> str:
        return "MISSING"

MISSING: _MissingSentinel = _MissingSentinel()


def _is_time_like(type_: Any) -> bool:
    """Return True when a type behaves like a timestamp or date."""
    if not isinstance(type_, type):
        return False
    if issubclass(type_, (datetime.datetime, datetime.date, datetime.time)):
        return True
    return bool(getattr(type_, "__time_like__", False) or getattr(type_, "__temporal__", False))


def _is_geometry_like(type_: Any) -> bool:
    """Return True when a type represents a geometry/location value."""
    if not isinstance(type_, type):
        return False
    return bool(
        getattr(type_, "__geometry__", False)
        or getattr(type_, "__geo__", False)
        or getattr(type_, "__geom__", False)
    )

# endregion

# =============================================================================
# Base Protodescriptor classes
# =============================================================================
# region Base Protodescriptor classes


@declarator
class Protodescriptor(Declarator):
    """Base class for all protodescriptors.

    Protodescriptors define attributes of primitive data types and models.
    They are called protodescriptors because they are not proper descriptors, but rather
    declarations that are used to generate descriptors in compiled types.
    """

    __reserved_patterns__ = {re.compile(r"^nx.*")}


@declarator
class AnnotatableDescriptor(Protodescriptor):
    """Base class for descriptors that can be annotated with type information."""
    annotation: str | None = field(init=False, default=None)  # Literal type annotation as a string
    type: "type | None" = field(init=False, default=None)  # Evaluated type annotation
    nullable: bool = field(init=False, default=None)
    __types__: ClassVar[tuple[type, ...]] = ()

    @abstractmethod
    def __validate_type__(self, type: Any) -> bool:
        return issubclass(type, self.__types__)

    def __set_type__(self, annotation: str, type: Any, nullable: bool):
        """Sets the type by supplying annotation (string), evaluated type, and nullability."""
        if type is not None and not self.__validate_type__(type):
            descriptor = self.name
            owner_name = self.owner.__name__
            msg = (
                f"Incompatible annotation {annotation} supplied to {descriptor} descriptor "
                + f"of {owner_name}."
            )
            raise TypeError(msg)
        self._set_once("annotation", annotation)
        self._set_once("type", type)
        self._set_once("nullable", nullable)


@declarator
class IdentifiableDescriptor(AnnotatableDescriptor):
    """Base class for descriptors of attributes that can uniquely identify an instance."""
    unique: bool = field(init=False, default=False)

    def __set_unique__(self, unique: bool):
        self._set_once("unique", unique)


@declarator
class AssignableDescriptor(IdentifiableDescriptor):
    """Base class for descriptors of attributes that receive their value through assignment."""
    default: Any = field(default=MISSING)
    factory: Callable[[], Any] | _MissingSentinel = field(default=MISSING)
    parser: Callable | Iterable[Callable] | None = field(default=None)
    validator: Callable | Iterable[Callable] | None = field(default=None)


@declarator
class CallableDescriptor(Protodescriptor):
    """A mixin class for descriptors that wrap callables."""
    callable: Callable | None = field(default=None)


@declarator
class FieldListDescriptor(Protodescriptor):
    """A mixin class for descriptors that reference a list of fields."""
    fields: tuple["FieldDescriptor", ...] = field(factory=tuple)
    admissible_field_types: ClassVar[tuple[type["FieldDescriptor"], ...]] = ()


@declarator
class MetaDescriptor(Protodescriptor):
    """Base class for prototype-level descriptors declared in archetypes and traits."""
    __handle__ = "meta"


@declarator
class FieldDescriptor(AnnotatableDescriptor):
    """Base class for descriptors that represent individual fields."""
    __handle__ = "field"
    load: str | None = field(default=None)
    repr: bool | Callable | str | None = field(default=None)


@declarator
class RelationDescriptor(FieldDescriptor):
    """Base descriptor for relation fields."""
    __handle__ = "relation"


@declarator
class MethodDescriptor(CallableDescriptor):
    """Base class for method descriptors."""
    __handle__ = "method"


@declarator
class ConstructionDescriptor(MethodDescriptor):
    """Base class for construction method descriptors."""
    __handle__ = "construction"


@declarator
class SchemaDescriptor(Protodescriptor):
    """Base class for descriptors that characterize schema features."""
    __handle__ = "schema"

# endregion

# =============================================================================
# Registry classes
# =============================================================================
# region Registry classes


class ProtodescriptorRegistry(DeclaratorRegistry[Protodescriptor]):
    """Base registry for protodescriptors."""
    pass


class RubrickedRegistry(Mapping):
    """A base class for registries organized by rubric."""
    __rubrics__: ClassVar[tuple[str, ...]] = ()  # Allowed rubrics in this registry

    def __init__(self):
        self._data: dict[str, dict[str, Any]] = {r: {} for r in self.__rubrics__}

    def __getitem__(self, rubric: str) -> dict[str, Any]:
        return self._data[rubric]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __register__(self, rubric: str, name: str, value: Any, overwrite: bool = False) -> None:
        """Registration primitive for an item under a given rubric and name."""
        if rubric not in self.__rubrics__:
            raise KeyError(f"Unknown rubric '{rubric}'")
        destination = self._data[rubric]
        if not overwrite and name in destination:
            raise KeyError(f"Duplicate entry '{name}' in rubric '{rubric}'")
        destination[name] = value

    @abstractmethod
    def register(self, item: Any, *, rubric: str, name: str, overwrite: bool = False) -> None:
        """Registers an item under a given rubric and name."""
        pass

    def rubrics(self, *, populated: bool = True) -> list[str]:
        """Returns the list of rubrics in the registry.

        If populated is set to True, only rubrics with at least one entry are returned.
        """
        if populated:
            return [r for r, bucket in self._data.items() if bucket]
        return list(self._data.keys())

    def flatten(self) -> Iterable[tuple[str, str, Any]]:
        for rubric, bucket in self._data.items():
            for name, value in bucket.items():
                yield rubric, name, value

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to a plain nested dict suitable for serialization."""
        return {rubric: dict(bucket) for rubric, bucket in self._data.items() if bucket}

    def to_yaml(self) -> str:
        """YAML-style pretty print."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def __str__(self) -> str:
        return self.to_yaml()
# endregion


# =============================================================================
# Concrete Protodescriptor classes
# =============================================================================
# region Concrete Protodescriptor classes


@declarator
class MetadataDescriptor(AssignableDescriptor, MetaDescriptor):
    """The descriptor for metadata fields."""
    __handle__ = "metadata"


@declarator
class OptionDescriptor(AssignableDescriptor, MetaDescriptor):
    """The descriptor for option fields."""
    __handle__ = "option"


@declarator
class NxFieldDescriptor(AssignableDescriptor, MetaDescriptor):
    """The descriptor for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"


@declarator
class DataDescriptor(AssignableDescriptor, FieldDescriptor):
    """The descriptor for data fields."""
    __handle__ = "data"
    index: bool | None = field(default=None)
    required: bool | None = field(default=None)
    typekey: bool | None = field(default=None)
    key: bool | None = field(default=None)
    sequence: bool | None = field(default=None)
    timestamp: bool | None = field(default=None)
    start_time: bool | None = field(default=None)
    end_time: bool | None = field(default=None)
    period: bool | None = field(default=None)
    location: bool | None = field(default=None)
    geom: bool | None = field(default=None)
    gt: Real | None = field(default=None)
    ge: Real | None = field(default=None)
    lt: Real | None = field(default=None)
    le: Real | None = field(default=None)
    min_length: int | None = field(default=None)
    max_length: int | None = field(default=None)
    pattern: str | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        temporal_flags = {
            "timestamp": self.timestamp is True,
            "start_time": self.start_time is True,
            "end_time": self.end_time is True,
            "period": self.period is True,
        }
        if sum(temporal_flags.values()) > 1:
            msg = "Temporal flags are mutually exclusive on data descriptors."
            raise ValueError(msg)
        if self.sequence is True and any(temporal_flags.values()):
            msg = "Sequence cannot be combined with temporal flags."
            raise ValueError(msg)
        if (self.start_time is True) != (self.end_time is True):
            msg = "start_time and end_time must be set together."
            raise ValueError(msg)

    def __set_type__(self, annotation: str, type: Any, nullable: bool):
        super().__set_type__(annotation, type, nullable)
        if (self.key is True or self.sequence is True) and nullable:
            msg = "Key and sequence fields must be non-nullable."
            raise TypeError(msg)
        if type is None:
            return
        if any(
            flag is True for flag in (self.timestamp, self.start_time, self.end_time, self.period)
        ) and not _is_time_like(type):
            msg = "Temporal flags require a time-like field type."
            raise TypeError(msg)
        if (self.location is True or self.geom is True) and not _is_geometry_like(type):
            msg = "Location/geom flags require a geometry-like field type."
            raise TypeError(msg)


@declarator
class LinkDescriptor(AssignableDescriptor, RelationDescriptor):
    on_delete: Literal["restrict", "set_null", "cascade"] = field(default="restrict")
    key: bool | None = field(default=None)
    __handle__ = "link"
    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        if self.on_delete not in {"restrict", "set_null", "cascade"}:
            msg = f"Invalid on_delete value {self.on_delete!r}."
            raise ValueError(msg)

    def __set_type__(self, annotation: str, type: Any, nullable: bool):
        super().__set_type__(annotation, type, nullable)
        if self.key is True and nullable:
            msg = "Key links must be non-nullable."
            raise TypeError(msg)


@declarator
class BackLinkDescriptor(IdentifiableDescriptor, RelationDescriptor):
    __handle__ = "backlink"
    via: type | None = field(default=None)
    limit: int | None = field(default=None)


@declarator
class SelectionDescriptor(RelationDescriptor, CallableDescriptor):
    __handle__ = "selection"
    kind: str | None = field(default=None)
    sql: Callable | None = field(default=None)
    ibis: Callable | None = field(default=None)
    fx: Callable | str | None = field(default=None)
    source: Any | None = field(default=None)
    key: Any | list[Any] | None = field(default=None)
    time: Any | None = field(default=None)
    space: Any | None = field(default=None)
    filter: Callable | Any | None = field(default=None)
    sort: str | list[str] | None = field(default=None)
    limit: int | None = field(default=None)


@declarator
class DocumentDescriptor(AssignableDescriptor, RelationDescriptor):
    __handle__ = "document"
    path: str | Any | None = field(default=None)
    format: str | None = field(default=None)
    compression: str | None = field(default=None)


@declarator
class FolderDescriptor(AssignableDescriptor, RelationDescriptor):
    __handle__ = "folder"
    path: str | Any | None = field(default=None)


@declarator
class StateDescriptor(RelationDescriptor, CallableDescriptor):
    __handle__ = "state"
    source: Any | None = field(default=None)
    reducer: str | Callable | None = field(default=None)
    max_lag: str | Any | None = field(default=None)
    min_observations: int | None = field(default=None)


@declarator
class FieldExpressionDescriptor(FieldDescriptor, CallableDescriptor):
    __handle__ = "fx"
    expr: Callable | str | None = field(default=None)


@declarator
class FieldGroupDescriptor(FieldDescriptor, FieldListDescriptor):
    __handle__ = "fieldgroup"


@declarator
class FieldBlockDescriptor(FieldDescriptor):
    __handle__ = "fieldblock"
    fields: tuple[str, ...] | None = field(default=None)


@declarator
class MetricDescriptor(FieldDescriptor, CallableDescriptor):
    __handle__ = "metric"
    expr: Callable | None = field(default=None)


@declarator
class ParserDescriptor(ConstructionDescriptor):
    __handle__ = "parser"
    fields: tuple[str, ...] | None = field(default=None)
    element_wise: bool = field(default=False)


@declarator
class ValidatorDescriptor(ConstructionDescriptor):
    __handle__ = "validator"
    fields: tuple[str, ...] | None = field(default=None)
    element_wise: bool = field(default=False)


@declarator
class KeyDescriptor(SchemaDescriptor, FieldListDescriptor):
    __handle__ = "key"
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )


@declarator
class SequenceDescriptor(SchemaDescriptor, FieldListDescriptor):
    __handle__ = "sequence"
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )


@declarator
class UnicityDescriptor(SchemaDescriptor, FieldListDescriptor):
    __handle__ = "unique"
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )


@declarator
class IndexDescriptor(SchemaDescriptor, FieldListDescriptor):
    __handle__ = "index"
    kind: str | None = field(default=None)
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )


@declarator
class PartitionDescriptor(SchemaDescriptor):
    __handle__ = "partition"
    key: Any | None = field(default=None)
    scheme: str | None = field(default=None)
    buckets: Any | None = field(default=None)


@declarator
class PathDescriptor(SchemaDescriptor):
    __handle__ = "path"
    template: str | None = field(default=None)


@declarator
class SortDescriptor(SchemaDescriptor):
    __handle__ = "sort"
    sortkeys: str | tuple[str, ...] | None = field(default=None)
# endregion
