"""This module defines the Protodescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import re
from collections.abc import Callable
from numbers import Real
from typing import Any, Literal, cast

from attrs import field

from anaximander.aml.metadescriptors import (
    MetadataDeclarator,
    MetadescriptorRegistry,
    NxFieldDeclarator,
    OptionDeclarator,
)

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
    declarative,
    declarator,
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
class FieldProtodescriptor(AnnotatableDeclarator, Protodescriptor):
    """Base class for descriptors that represent individual fields."""
    __handle__ = "field"
    load: str | None = field(default=None)
    repr: bool | Callable | str | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("load", self.load, str, allow_none=True)
        self._validate_value_type("repr", self.repr, (bool, Callable, str), allow_none=True)  # type: ignore[arg-type]


@declarator
class AssignableFieldProtodescriptor(AssignableDeclarator, FieldProtodescriptor):
    """Abstract base class for assignable field descriptors ('data' and 'link')."""
    pass


@declarator
class RelationProtodescriptor(FieldProtodescriptor):
    """Base descriptor for relation fields."""
    __handle__ = "relation"


@declarator
class ConstructionDeclarator(CallableDeclarator[Callable[[declarative, Any], bool]], Protodescriptor):  # noqa
    """Base class for construction method descriptors."""
    __handle__ = "construction"
    callable: Callable[[declarative, Any], bool] = field()


@declarator
class SchemaDeclarator(Declarator):
    """Base class for declarators that characterize schema features."""
    __handle__ = "schema"


@declarator
class FieldEnumeration(EnumerationDeclarator):
    """Base class for enumerations of field protodescriptors."""
    __member_types__ = (FieldProtodescriptor,)


@declarator
class AssignableFieldEnumeration(EnumerationDeclarator):
    """Base class for enumerations of assignable field protodescriptors."""
    __member_types__ = (AssignableFieldProtodescriptor,)


# endregion

# =============================================================================
# Concrete Protodescriptor classes
# =============================================================================
# region Concrete Protodescriptor classes


@declarator
class DataProtodescriptor(AssignableFieldProtodescriptor):
    """The protodescriptor class for data fields."""
    __handle__ = "data"
    index: bool | None = field(default=None)
    required: bool | None = field(default=False)
    typekey: bool | None = field(default=False)
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

    def __validate__(self) -> None:
        super().__validate__()
        for attr in ("index", "required", "typekey", "key", "sequence", "timestamp",
                     "start_time", "end_time", "period", "location", "geom"):
            self._validate_value_type(attr, getattr(self, attr), bool, allow_none=True)
        for attr in ("gt", "ge", "lt", "le"):
            self._validate_value_type(attr, getattr(self, attr), Real, allow_none=True)
        self._validate_value_type("min_length", self.min_length, int, allow_none=True)
        self._validate_value_type("max_length", self.max_length, int, allow_none=True)
        self._validate_value_type("pattern", self.pattern, str, allow_none=True)


@declarator
class LinkProtodescriptor(AssignableFieldProtodescriptor, RelationProtodescriptor):
    """The protodescriptor class for link fields."""
    on_delete: Literal["restrict", "set_null", "cascade"] = field(default="restrict")
    key: bool | None = field(default=None)
    __handle__ = "link"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("on_delete", self.on_delete, str)
        self._validate_value_type("key", self.key, bool, allow_none=True)


@declarator
class BackLinkProtodescriptor(IdentifiableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for backlink fields."""
    __handle__ = "backlink"
    via: type | None = field(default=None)
    limit: int | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("via", self.via, type, allow_none=True)
        self._validate_value_type("limit", self.limit, int, allow_none=True)


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

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("kind", self.kind, str, allow_none=True)
        self._validate_value_type("sql", self.sql, Callable, allow_none=True)  # type: ignore[arg-type]
        self._validate_value_type("ibis", self.ibis, Callable, allow_none=True)  # type: ignore[arg-type]
        self._validate_value_type("fx", self.fx, (Callable, str), allow_none=True)  # type: ignore[arg-type]
        if self.sort is not None:
            if isinstance(self.sort, list):
                if any(not isinstance(item, str) for item in self.sort):
                    raise TypeError("SelectionProtodescriptor.sort must contain only strings.")
            elif not isinstance(self.sort, str):
                raise TypeError("SelectionProtodescriptor.sort must be a string or list of strings.")  # noqa
        self._validate_value_type("limit", self.limit, int, allow_none=True)


@declarator
class DocumentProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for document fields."""
    __handle__ = "document"
    path: str | Any | None = field(default=None)
    format: str | None = field(default=None)
    compression: str | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("format", self.format, str, allow_none=True)
        self._validate_value_type("compression", self.compression, str, allow_none=True)


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

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("reducer", self.reducer, (Callable, str), allow_none=True)  # type: ignore[arg-type]
        self._validate_value_type("min_observations", self.min_observations, int, allow_none=True)


@declarator
class FieldExpressionProtodescriptor(FieldProtodescriptor, CallableDeclarator):
    """The protodescriptor class for field expressions."""
    __handle__ = "fx"
    ref: str | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("ref", self.ref, str, allow_none=True)


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
    element_wise: bool | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("element_wise", self.element_wise, bool, allow_none=True)


@declarator
class ValidatorDeclarator(ConstructionDeclarator, FieldEnumeration):
    """The declarator class for field validators."""
    __handle__ = "validator"
    element_wise: bool | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("element_wise", self.element_wise, bool, allow_none=True)


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

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("kind", self.kind, str, allow_none=True)


@declarator
class PartitionDeclarator(SchemaDeclarator):
    """The declarator class for schema partitions."""
    __handle__ = "partition"
    key: Any | None = field(default=None)
    scheme: str | None = field(default=None)
    buckets: Any | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("scheme", self.scheme, str, allow_none=True)


@declarator
class PathDeclarator(SchemaDeclarator):
    """The declarator class for traversal paths."""
    __handle__ = "path"
    template: str | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("template", self.template, str, allow_none=True)


@declarator
class SortDeclarator(FieldEnumeration, SchemaDeclarator):
    """The declarator class for schema sort orders."""
    __handle__ = "sort"
    sort_directions: list[Literal["asc", "desc"]] | None = field(default=None)

    def __validate__(self) -> None:
        super().__validate__()
        if self.sort_directions is None:
            return
        if not isinstance(self.sort_directions, list):
            raise TypeError("SortDeclarator.sort_directions must be a list of 'asc'/'desc'.")
        invalid = [v for v in self.sort_directions if v not in {"asc", "desc"}]
        if invalid:
            raise TypeError("SortDeclarator.sort_directions must be 'asc' or 'desc'.")

# endregion

# =============================================================================
# Registry class
# =============================================================================
# region Registry class


class ProtodescriptorRegistry(MultiRegistry):
    """Registry for meta bindings and protodescriptors."""

    __namespaces__ = {"metadata", "nxfield", "option", "field", "schema", "construction", "data"}

    def __init__(self, metadescriptors: MetadescriptorRegistry):
        metadata = BindingRegistry(_declarators=metadescriptors.metadata)
        nxfield = BindingRegistry(_declarators=metadescriptors.nxfield)
        option = BindingRegistry(_declarators=metadescriptors.option)
        field = DeclaratorRegistry[FieldProtodescriptor]()
        schema = DeclaratorRegistry[SchemaDeclarator]()
        construction = DeclaratorRegistry[ConstructionDeclarator]()
        data = BindingRegistry(_declarators=field)
        super().__init__(
            metadata=metadata,
            nxfield=nxfield,
            option=option,
            field=field,
            schema=schema,
            construction=construction,
            data=data,
        )
        self._metadescriptors = metadescriptors

    @property
    def metadescriptors(self) -> MetadescriptorRegistry | None:
        """Returns the metadescriptor registry, or None if it has been garbage collected."""
        return self._metadescriptors

    @property
    def metadata(self) -> BindingRegistry[MetadataDeclarator]:
        """Returns the metadata binding registry."""
        return cast(BindingRegistry[MetadataDeclarator], self._data["metadata"])

    @property
    def nxfield(self) -> BindingRegistry[NxFieldDeclarator]:
        """Returns the nxfield binding registry."""
        return cast(BindingRegistry[NxFieldDeclarator], self._data["nxfield"])

    @property
    def option(self) -> BindingRegistry[OptionDeclarator]:
        """Returns the option binding registry."""
        return cast(BindingRegistry[OptionDeclarator], self._data["option"])

    @property
    def field(self) -> DeclaratorRegistry[FieldProtodescriptor]:
        """Returns the field protodescriptor registry."""
        return cast(DeclaratorRegistry[FieldProtodescriptor], self._data["field"])

    @property
    def schema(self) -> DeclaratorRegistry[SchemaDeclarator]:
        """Returns the schema declarator registry."""
        return cast(DeclaratorRegistry[SchemaDeclarator], self._data["schema"])

    @property
    def construction(self) -> DeclaratorRegistry[ConstructionDeclarator]:
        """Returns the construction declarator registry."""
        return cast(DeclaratorRegistry[ConstructionDeclarator], self._data["construction"])

    @property
    def data(self) -> BindingRegistry[DataProtodescriptor]:
        """Returns the data binding registry."""
        return cast(BindingRegistry[DataProtodescriptor], self._data["data"])

    def register(self, key, item: Any, *, namespace: str | None = None) -> None:
        """Registers a declarator or binding in the appropriate registry."""
        # If the namespace is specified, use it directly
        if namespace is not None:
            registry = self.get_registry(namespace)
            registry.register(key, item)
            return
        # If the item is a declarator, route to the appropriate registry
        match = True
        match item:
            case FieldProtodescriptor():
                self.field.register(key, item)
            case SchemaDeclarator():
                self.schema.register(key, item)
            case ConstructionDeclarator():
                self.construction.register(key, item)
            case _:
                match = False
        if match:
            return
        # If the item is a binding, try to register in each binding registry
        for binding_registry in (self.metadata, self.nxfield, self.option, self.data):
            try:
                binding_registry.register(key, item)
                return
            except KeyError:
                continue
        raise KeyError(f"Cannot register item {item} with key {key} in any registry.")

# endregion
