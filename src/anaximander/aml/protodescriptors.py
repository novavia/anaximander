"""This module defines the Protodescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import datetime
import re
import weakref
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

from .declarative import (
    AnnotatableDeclarator,
    AssignableDeclarator,
    BindingRegistry,
    CallableDeclarator,
    Declarator,
    DeclaratorRegistry,
    EnumerationDeclarator,
    IdentifiableDeclarator,
    MultiRegistry,
    declarator,
)

# endregion

# =============================================================================
# Constants and Utilities
# =============================================================================
# region Constants and Utilities


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
    __handle__ = "proto"
    __reserved_patterns__ = {re.compile(r"^nx.*")}


@declarator
class Metadescriptor(Declarator):
    """Base class for prototype-level descriptors declared in archetypes and traits."""
    __handle__ = "meta"
    __reserved_patterns__ = {re.compile(r"^nx.*")}


@declarator
class FieldProtodescriptor(AnnotatableDeclarator, Protodescriptor):
    """Base class for descriptors that represent individual fields."""
    __handle__ = "field"
    load: str | None = field(default=None)
    repr: bool | Callable | str | None = field(default=None)


@declarator
class AssignableFieldProtodescriptor(AssignableDeclarator, FieldProtodescriptor):
    """Abstract base class for assignable field descriptors ('data' and 'link')."""
    pass


@declarator
class RelationProtodescriptor(FieldProtodescriptor):
    """Base descriptor for relation fields."""
    __handle__ = "relation"


@declarator
class ConstructionDeclarator(CallableDeclarator, Protodescriptor):
    """Base class for construction method descriptors."""
    __handle__ = "construction"


@declarator
class SchemaDeclarator(Metadescriptor):
    """Base class for declarators that characterize schema features."""
    __handle__ = "schema"


@declarator
class FieldEnumeration(EnumerationDeclarator[FieldProtodescriptor]):
    """Base class for enumerations of field protodescriptors."""
    __member_types__ = (FieldProtodescriptor,)


@declarator
class AssignableFieldEnumeration(EnumerationDeclarator[AssignableFieldProtodescriptor]):
    """Base class for enumerations of assignable field protodescriptors."""
    __member_types__ = (AssignableFieldProtodescriptor,)


# endregion

# =============================================================================
# Concrete Protodescriptor classes
# =============================================================================
# region Concrete Protodescriptor classes


@declarator
class MetadataDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for metadata fields."""
    __handle__ = "metadata"


@declarator
class OptionDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for option fields."""
    __handle__ = "option"


@declarator
class NxFieldDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"


@declarator
class MetadataValidator(CallableDeclarator, Metadescriptor):
    """The metadescriptor class for metadata validators."""
    __handle__ = "metavalidator"


@declarator
class DataProtodescriptor(AssignableFieldProtodescriptor):
    """The protodescriptor class for data fields."""
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
class LinkProtodescriptor(AssignableFieldProtodescriptor, RelationProtodescriptor):
    """The protodescriptor class for link fields."""
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
class BackLinkProtodescriptor(IdentifiableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for backlink fields."""
    __handle__ = "backlink"
    via: type | None = field(default=None)
    limit: int | None = field(default=None)


@declarator
class SelectionProtodescriptor(RelationProtodescriptor, CallableDeclarator):
    """The protodescriptor class for selection relations."""
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
class DocumentProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for document fields."""
    __handle__ = "document"
    path: str | Any | None = field(default=None)
    format: str | None = field(default=None)
    compression: str | None = field(default=None)


@declarator
class FolderProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for folder fields."""
    __handle__ = "folder"
    path: str | Any | None = field(default=None)


@declarator
class StateProtodescriptor(RelationProtodescriptor, CallableDeclarator):
    """The protodescriptor class for state fields."""
    __handle__ = "state"
    source: Any | None = field(default=None)
    reducer: str | Callable | None = field(default=None)
    max_lag: str | Any | None = field(default=None)
    min_observations: int | None = field(default=None)


@declarator
class FieldExpressionProtodescriptor(FieldProtodescriptor, CallableDeclarator):
    """The protodescriptor class for field expressions."""
    __handle__ = "fx"
    ref: str | None = field(default=None)


@declarator
class FieldGroupProtodescriptor(FieldProtodescriptor, FieldEnumeration):
    """The protodescriptor class for field groups."""
    __handle__ = "fieldgroup"


@declarator
class FieldBlockProtodescriptor(FieldProtodescriptor, FieldEnumeration):
    """The protodescriptor class for field blocks."""
    __handle__ = "fieldblock"


@declarator
class MetricProtodescriptor(FieldProtodescriptor, CallableDeclarator):
    """The protodescriptor class for metric fields."""
    __handle__ = "metric"


@declarator
class ParserDeclarator(ConstructionDeclarator, FieldEnumeration):
    """The declarator class for field parsers."""
    __handle__ = "parser"
    element_wise: bool = field(default=False)


@declarator
class ValidatorDeclarator(ConstructionDeclarator, FieldEnumeration):
    """The declarator class for field validators."""
    __handle__ = "validator"
    element_wise: bool = field(default=False)


@declarator
class KeyDeclarator(AssignableFieldEnumeration, SchemaDeclarator):
    """The declarator class for schema keys."""
    __handle__ = "key"


@declarator
class SequenceDeclarator(AssignableFieldEnumeration, SchemaDeclarator):
    """The declarator class for schema sequences."""
    __handle__ = "sequence"


@declarator
class UnicityDeclarator(AssignableFieldEnumeration, SchemaDeclarator):
    """The declarator class for schema unique constraints."""
    __handle__ = "unique"


@declarator
class IndexDeclarator(AssignableFieldEnumeration, SchemaDeclarator):
    """The declarator class for schema indexes."""
    __handle__ = "index"
    kind: str | None = field(default=None)


@declarator
class PartitionDeclarator(SchemaDeclarator):
    """The declarator class for schema partitions."""
    __handle__ = "partition"
    key: Any | None = field(default=None)
    scheme: str | None = field(default=None)
    buckets: Any | None = field(default=None)


@declarator
class PathDeclarator(SchemaDeclarator):
    """The declarator class for traversal paths."""
    __handle__ = "path"
    template: str | None = field(default=None)


@declarator
class SortDeclarator(FieldEnumeration, SchemaDeclarator):
    """The declarator class for schema sort orders."""
    __handle__ = "sort"
    sort_directions: list[Literal["asc", "desc"]] | None = field(default=None)

# endregion

# =============================================================================
# Registry classes
# =============================================================================
# region Registry classes


class MetadescriptorRegistry(MultiRegistry):
    """Registry for metadescriptors."""
    metadata: DeclaratorRegistry[MetadataDeclarator]
    nxfield: DeclaratorRegistry[NxFieldDeclarator]
    option: DeclaratorRegistry[OptionDeclarator]

    def __init__(self):
        self.metadata = DeclaratorRegistry[MetadataDeclarator]()
        self.nxfield = DeclaratorRegistry[NxFieldDeclarator]()
        self.option = DeclaratorRegistry[OptionDeclarator]()
        super().__init__(metadata=self.metadata, nxfield=self.nxfield, option=self.option)


class ProtodescriptorRegistry(MultiRegistry):
    """Registry for meta bindings and protodescriptors."""
    metadata: BindingRegistry[MetadataDeclarator]
    nxfield: BindingRegistry[NxFieldDeclarator]
    option: BindingRegistry[OptionDeclarator]
    field: DeclaratorRegistry[FieldProtodescriptor]
    schema: DeclaratorRegistry[SchemaDeclarator]
    construction: DeclaratorRegistry[ConstructionDeclarator]
    bindings: BindingRegistry[FieldProtodescriptor]

    def __init__(self, metadescriptors: MetadescriptorRegistry):
        self.metadata = metadescriptors.metadata.bindings
        self.nxfield = metadescriptors.nxfield.bindings
        self.option = metadescriptors.option.bindings
        self.field = DeclaratorRegistry[FieldProtodescriptor]()
        self.schema = DeclaratorRegistry[SchemaDeclarator]()
        self.construction = DeclaratorRegistry[ConstructionDeclarator]()
        self.bindings = self.field.bindings
        super().__init__(
            metadata=self.metadata,
            nxfield=self.nxfield,
            option=self.option,
            field=self.field,
            schema=self.schema,
            construction=self.construction,
            bindings=self.bindings,
        )
        self._metadescriptors_ref = (weakref.ref(metadescriptors))

    @property
    def metadescriptors(self) -> MetadescriptorRegistry | None:
        """Returns the metadescriptor registry, or None if it has been garbage collected."""
        return self._metadescriptors_ref()

# endregion
