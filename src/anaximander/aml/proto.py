"""This module defines the Protodescriptor class for the Anaximander Modeling Language (AML)."""

import ast
import datetime
import re
from abc import ABC, ABCMeta, abstractmethod
from numbers import Real
from types import MappingProxyType, NoneType
from typing import (
    Callable,
    ClassVar,
    Iterable,
    Literal,
    Protocol,
    Mapping,
    Any,
    TypeVar,
    get_args,
    get_type_hints,
    overload,
)

import attrs
from annotationlib import Format, get_annotations

from anaximander.utils.funcs import type_name_to_collection_name

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

type Assignment = ast.Assign | ast.AnnAssign

# from pydantic import BaseModel   # pydantic not yet compatible with python 3.14


class DescriptorConfig(Protocol):
    """A protocol for descriptor configuration."""
    def resolve(self, **context) -> Mapping[str, Any]: ...


type ConfigValue = Any | DescriptorConfig | Mapping[str, ConfigValue]
type Config = DescriptorConfig | Mapping[str, ConfigValue]


@attrs.define(frozen=True)
class Protodescriptor(ABC):
    """Base class for all protodescriptors.

    Protodescriptors define attributes of primitive data types and models.
    They are called protodescriptors because they are not proper descriptors, but rather
    declarations that are used to generate descriptors in compiled types.
    """

    __reserved_patterns__: ClassVar[set[re.Pattern[str]]] = {
        re.compile(r"^nx.*"),
        re.compile(r"^__.*"),
    }

    # Post-init wired fields (logically immutable; set via internal backdoor).
    name: str = attrs.field(init=False, default=None)
    owner: "Prototype" = attrs.field(init=False, default=None)
    __ast__: ast.AST = attrs.field(init=False, default=None)

    # Init-time fields (immutable)
    doc: str | None = attrs.field(default=None)
    config: Mapping[str, ConfigValue] = attrs.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        # Freeze config to prevent accidental mutation.
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    def __init_subclass__(cls):
        super().__init_subclass__()
        parent = cls.mro()[1]
        reserved_patterns: set[re.Pattern[str]] = getattr(parent, "__reserved_patterns__", set())
        if "__reserved_patterns__" in vars(cls):
            try:
                assert all(isinstance(pattern, re.Pattern) for pattern in cls.__reserved_patterns__)
            except AssertionError:
                raise TypeError("All elements of __reserved_patterns__ must be instances of re.Pattern")
            cls.__reserved_patterns__ = reserved_patterns | set(cls.__reserved_patterns__)
        else:
            cls.__reserved_patterns__ = reserved_patterns

    def _set_once(self, attr: str, value: Any) -> None:
        """Internal backdoor: set a frozen attribute once (or idempotently)."""
        current = getattr(self, attr)
        if current is not None and current != value:
            raise RuntimeError(f"{self.__class__.__name__}.{attr} is already set.")
        object.__setattr__(self, attr, value)

    def __set_name__(self, owner: "Prototype", name: str):
        if any(pattern.fullmatch(name) for pattern in self.__reserved_patterns__):
            mdtype = self.__class__.__name__
            msg = f"Cannot use reserved name {name} for Protodescriptor or type {mdtype}."
            raise ValueError(msg)
        self._set_once("name", name)
        self._set_once("owner", owner)

    def __set_ast__(self, node: ast.AST | None) -> None:
        """Attach the AST node that declared this protodescriptor (if any)."""
        self._set_once("__ast__", node)


P = TypeVar("P", bound=Protodescriptor)


class Prototype(ABCMeta):
    """Metaclass for model and data declarative types."""
    __compilations__: dict[str, dict]  # Holds compilation target handles and parameters
    __ast__: ast.ClassDef  # Holds the model's parsed abstract syntax tree

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__compilations__: dict[str, dict] = {}

    @overload
    def protodescriptors(cls, *, inherited: bool = True) -> Mapping[str, Protodescriptor]: ...
    @overload
    def protodescriptors(cls, *types: type[P], inherited: bool = True) -> Mapping[str, P]: ...

    def protodescriptors(
        cls,
        *types: type[Protodescriptor],
        inherited: bool = True,
    ) -> Mapping[str, Protodescriptor]:
        """Returns a dictionary of protodescriptors of the supplied types.

        If inherited is set to True, protodescriptors declared in parent models are included.
        Otherwise, only the protodescriptors directly declared by cls are returned.
        """
        selected_types: tuple[type[Protodescriptor], ...] = types or (Protodescriptor,)
        cls_protodescriptors = {
            k: v for k, v in cls.__dict__.items() if isinstance(v, selected_types)
        }
        if inherited:
            parent = cls.mro()[1]
            if isinstance(parent, Prototype):
                parent_protodescriptors = dict(parent.protodescriptors(*selected_types, inherited=True))
                protodescriptors = parent_protodescriptors | cls_protodescriptors
            else:
                protodescriptors = cls_protodescriptors
        else:
            protodescriptors = cls_protodescriptors
        return protodescriptors

    def __validate_bases__(cls):
        """The first base must be a Prototype and there can be only one."""
        bases = cls.__bases__
        if not isinstance(bases[0], Prototype):
            msg = "Prototypes cannot be used as mixin classes."
            raise TypeError(msg)
        extra_parent_prototypes = [b for b in bases[1:] if isinstance(b, Prototype)]
        if extra_parent_prototypes:
            msg = "Prototypes do not support multiple inheritance."
            raise TypeError(msg)

    @staticmethod
    def __unwrap_optional_type__(type_hint: Any) -> tuple[Any, bool]:
        """Detect optional annotations and return base type with a nullable flag."""
        if type_hint is None:
            return None, False
        if type_hint is NoneType:
            return None, True
        args = get_args(type_hint)
        if args and any(arg is NoneType for arg in args):
            non_none_args = [arg for arg in args if arg is not NoneType]
            base_type = non_none_args[0] if non_none_args else None
            return base_type, True
        return type_hint, False

    def __set_type_annotations__(cls):
        """Sets type annotations on typed protodescriptors."""
        annotation_values = get_annotations(cls, format=Format.VALUE)
        annotation_strings = get_annotations(cls, format=Format.STRING)
        superhints = get_type_hints(cls)  # This includes super classes
        protodescriptors = cls.protodescriptors(AnnotatableDescriptor, inherited=False)
        for name, protodescriptor in protodescriptors.items():
            annotation_value = annotation_values.get(name, "")
            annotation_string = annotation_strings.get(name, "")
            if isinstance(annotation_value, str):
                annotation = f'"{annotation_value}"'
            else:
                annotation = annotation_string
            # TODO: normalize to a prototype in case of model attributes
            # TODO: normalize to a metadata type in case of option / metacharacter
            hint = superhints.get(name, None)
            hint, nullable = cls.__unwrap_optional_type__(hint)
            protodescriptor.__set_type__(annotation, hint, nullable)

    @property
    def collection_name(cls):
        """A collection name using camel case and pluralization.

        This can be customized by passing metadata (#TODO).
        """
        return type_name_to_collection_name(cls.__name__)
    

@attrs.define(frozen=True)
class AnnotatableDescriptor(Protodescriptor):
    """Base class for descriptors that can be annotated with type information."""
    annotation: str | None = attrs.field(init=False, default=None)  # Literal type annotation as a string
    type: type | None = attrs.field(init=False, default=None)  # Evaluated type annotation
    nullable: bool = attrs.field(init=False, default=None)
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


@attrs.define(frozen=True)
class IdentifiableDescriptor(AnnotatableDescriptor):
    """Base class for descriptors of attributes that can uniquely identify an instance."""
    unique: bool = attrs.field(init=False, default=False)

    def __set_unique__(self, unique: bool):
        self._set_once("unique", unique)


@attrs.define(frozen=True)
class AssignableDescriptor(IdentifiableDescriptor):
    """Base class for descriptors of attributes that receive their value through assignment."""
    default: Any = attrs.field(default=MISSING)
    factory: Callable[[], Any] | _MissingSentinel = attrs.field(default=MISSING)
    parser: Callable | Iterable[Callable] | None = attrs.field(default=None)
    validator: Callable | Iterable[Callable] | None = attrs.field(default=None)


@attrs.define(frozen=True)
class CallableDescriptor(Protodescriptor):
    """A mixin class for descriptors that wrap callables."""
    callable: Callable | None = attrs.field(default=None)


@attrs.define(frozen=True)
class FieldListDescriptor(Protodescriptor):
    """A mixin class for descriptors that reference a list of fields."""
    fields: tuple["FieldDescriptor", ...] = attrs.field(factory=tuple)
    admissible_field_types: ClassVar[tuple[type["FieldDescriptor"], ...]] = ()


@attrs.define(frozen=True)
class MetaDescriptor(AssignableDescriptor):
    """Base class for descriptors that target the nx inner class of archetypes and traits."""
    pass

@attrs.define(frozen=True)
class FieldDescriptor(AnnotatableDescriptor):
    """Base class for descriptors that represent individual fields."""
    load: str | None = attrs.field(default=None)
    repr: bool | Callable | str | None = attrs.field(default=None)

@attrs.define(frozen=True)
class RelationDescriptor(FieldDescriptor):
    """Base descriptor for relation fields."""
    pass

@attrs.define(frozen=True)
class MethodDescriptor(CallableDescriptor):
    """Base class for method descriptors."""
    pass

@attrs.define(frozen=True)
class ConstructionDescriptor(MethodDescriptor):
    """Base class for construction method descriptors."""
    pass

@attrs.define(frozen=True)
class SchemaDescriptor(Protodescriptor):
    """Base class for descriptors that characterize schema features."""
    pass

# =============================================================================
# Concrete Protodescriptor classes
# =============================================================================


@attrs.define(frozen=True)
class MetaCharacter(MetaDescriptor):
    pass


@attrs.define(frozen=True)
class OptionDescriptor(MetaDescriptor):
    pass


@attrs.define(frozen=True)
class NxFieldDescriptor(MetaDescriptor):
    pass


@attrs.define(frozen=True)
class DataDescriptor(AssignableDescriptor, FieldDescriptor):
    """The descriptor for data fields."""
    index: bool | None = attrs.field(default=None)
    required: bool | None = attrs.field(default=None)
    typekey: bool | None = attrs.field(default=None)
    key: bool | None = attrs.field(default=None)
    sequence: bool | None = attrs.field(default=None)
    timestamp: bool | None = attrs.field(default=None)
    start_time: bool | None = attrs.field(default=None)
    end_time: bool | None = attrs.field(default=None)
    period: bool | None = attrs.field(default=None)
    location: bool | None = attrs.field(default=None)
    geom: bool | None = attrs.field(default=None)
    gt: Real | None = attrs.field(default=None)
    ge: Real | None = attrs.field(default=None)
    lt: Real | None = attrs.field(default=None)
    le: Real | None = attrs.field(default=None)
    min_length: int | None = attrs.field(default=None)
    max_length: int | None = attrs.field(default=None)
    pattern: str | None = attrs.field(default=None)

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

@attrs.define(frozen=True)
class LinkDescriptor(AssignableDescriptor, RelationDescriptor):
    on_delete: Literal["restrict", "set_null", "cascade"] = attrs.field(default="restrict")
    key: bool | None = attrs.field(default=None)

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

@attrs.define(frozen=True)
class BackLinkDescriptor(IdentifiableDescriptor, RelationDescriptor):
    via: type | None = attrs.field(default=None)
    limit: int | None = attrs.field(default=None)

@attrs.define(frozen=True)
class SelectionDescriptor(RelationDescriptor, CallableDescriptor):
    kind: str | None = attrs.field(default=None)
    sql: Callable | None = attrs.field(default=None)
    ibis: Callable | None = attrs.field(default=None)
    fx: Callable | str | None = attrs.field(default=None)
    source: Any | None = attrs.field(default=None)
    key: Any | list[Any] | None = attrs.field(default=None)
    time: Any | None = attrs.field(default=None)
    space: Any | None = attrs.field(default=None)
    filter: Callable | Any | None = attrs.field(default=None)
    sort: str | list[str] | None = attrs.field(default=None)
    limit: int | None = attrs.field(default=None)

@attrs.define(frozen=True)
class DocumentDescriptor(AssignableDescriptor, RelationDescriptor):
    path: str | Any | None = attrs.field(default=None)
    format: str | None = attrs.field(default=None)
    compression: str | None = attrs.field(default=None)

@attrs.define(frozen=True)
class FolderDescriptor(AssignableDescriptor, RelationDescriptor):
    path: str | Any | None = attrs.field(default=None)

@attrs.define(frozen=True)
class StateDescriptor(RelationDescriptor, CallableDescriptor):
    source: Any | None = attrs.field(default=None)
    reducer: str | Callable | None = attrs.field(default=None)
    max_lag: str | Any | None = attrs.field(default=None)
    min_observations: int | None = attrs.field(default=None)

@attrs.define(frozen=True)
class FieldExpressionDescriptor(FieldDescriptor, CallableDescriptor):
    expr: Callable | str | None = attrs.field(default=None)

@attrs.define(frozen=True)
class FieldGroupDescriptor(FieldDescriptor, FieldListDescriptor):
    pass

@attrs.define(frozen=True)
class FieldBlockDescriptor(FieldDescriptor):
    fields: tuple[str, ...] | None = attrs.field(default=None)

@attrs.define(frozen=True)
class MetricDescriptor(FieldDescriptor, CallableDescriptor):
    expr: Callable | None = attrs.field(default=None)

@attrs.define(frozen=True)
class ParserDescriptor(ConstructionDescriptor):
    fields: tuple[str, ...] | None = attrs.field(default=None)
    element_wise: bool = attrs.field(default=False)

@attrs.define(frozen=True)
class ValidatorDescriptor(ConstructionDescriptor):
    fields: tuple[str, ...] | None = attrs.field(default=None)
    element_wise: bool = attrs.field(default=False)

@attrs.define(frozen=True)
class KeyDescriptor(SchemaDescriptor, FieldListDescriptor):
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )

@attrs.define(frozen=True)
class SequenceDescriptor(SchemaDescriptor, FieldListDescriptor):
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )

@attrs.define(frozen=True)
class UnicityDescriptor(SchemaDescriptor, FieldListDescriptor):
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )

@attrs.define(frozen=True)
class IndexDescriptor(SchemaDescriptor, FieldListDescriptor):
    kind: str | None = attrs.field(default=None)
    admissible_field_types: ClassVar[tuple[type[FieldDescriptor], ...]] = (
        DataDescriptor,
        LinkDescriptor,
    )

@attrs.define(frozen=True)
class PartitioningDescriptor(SchemaDescriptor):
    key: Any | None = attrs.field(default=None)
    scheme: str | None = attrs.field(default=None)
    buckets: Any | None = attrs.field(default=None)

@attrs.define(frozen=True)
class PathDescriptor(SchemaDescriptor):
    template: str | None = attrs.field(default=None)

@attrs.define(frozen=True)
class SortDescriptor(SchemaDescriptor):
    sortkeys: str | tuple[str, ...] | None = attrs.field(default=None)
