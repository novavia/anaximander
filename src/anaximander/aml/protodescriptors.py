"""This module defines the Protodescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import re
from collections.abc import Callable
from datetime import timedelta
from numbers import Real
from typing import Any, Literal, Self, cast

from attrs import field

from .declarative import (
    MISSING,
    AnnotatableDeclarator,
    AssignableDeclarator,
    BindingRegistry,
    CallableDeclarator,
    Declarator,
    DeclaratorRegistry,
    EnumerationDeclarator,
    IdentifiableDeclarator,
    Missing,
    MultiRegistry,
    _tighten_bool,
    _tighten_classvar,
    _tighten_nullable,
    _tighten_type,
    declarative,
    declarator,
    is_missing,
    is_not_missing,
)
from .metadescriptors import (
    MetadataDeclarator,
    MetadescriptorRegistry,
    NxFieldDeclarator,
    OptionDeclarator,
)

# endregion

# =============================================================================
# Base Protodescriptor classes
# =============================================================================
# region Base Protodescriptor classes


def _lower_bound(gt: Real | Missing, ge: Real | Missing) -> tuple[Real, bool] | None:
    """Resolve lower-bound value and strictness from gt/ge constraints."""
    if is_not_missing(gt):
        return gt, True
    if is_not_missing(ge):
        return ge, False
    return None


def _upper_bound(lt: Real | Missing, le: Real | Missing) -> tuple[Real, bool] | None:
    """Resolve upper-bound value and strictness from lt/le constraints."""
    if is_not_missing(lt):
        return lt, True
    if is_not_missing(le):
        return le, False
    return None


def _tighten_lower(base_gt: Real | Missing, base_ge: Real | Missing,
                   new_gt: Real | Missing, new_ge: Real | Missing) -> bool:
    """Whether the lower-bound constraint is monotonically tightened."""
    base = _lower_bound(base_gt, base_ge)
    new = _lower_bound(new_gt, new_ge)
    if base is None or new is None:
        return True
    bval, bstrict = base
    nval, nstrict = new
    if nval < bval:
        return False
    if nval == bval and bstrict and not nstrict:
        return False
    return True


def _tighten_upper(base_lt: Real | Missing, base_le: Real | Missing,
                   new_lt: Real | Missing, new_le: Real | Missing) -> bool:
    """Whether the upper-bound constraint is monotonically tightened."""
    base = _upper_bound(base_lt, base_le)
    new = _upper_bound(new_lt, new_le)
    if base is None or new is None:
        return True
    bval, bstrict = base
    nval, nstrict = new
    if nval > bval:
        return False
    if nval == bval and bstrict and not nstrict:
        return False
    return True


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
    load: Literal["eager", "lazy"] | Missing = field(default=MISSING)
    repr: bool | Callable | str | Missing = field(default=MISSING)

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values."""
        return bool(self.classvar)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.load):
            if self.load not in {"eager", "lazy"}:
                raise TypeError("Field protodescriptor's load attribute must be 'eager' or 'lazy'.")  # noqa
        if is_not_missing(self.repr):
            self._validate_value_type("repr", self.repr, (bool, Callable, str))  # type: ignore[arg-type]

    def _validate_override_common(self, override: "FieldProtodescriptor") -> None:
        """Validate shared tightening rules for field protodescriptors."""
        if not _tighten_type(self.type, override.type):
            raise AttributeError("Field protodescriptor type cannot be loosened.")
        if not _tighten_nullable(self.nullable, override.nullable):
            raise AttributeError("Field protodescriptor nullability cannot be loosened.")
        if not _tighten_classvar(self.classvar, override.classvar):
            raise AttributeError("Field protodescriptor classvar cannot be overridden.")
        if is_not_missing(self.load) and is_not_missing(override.load):
            if self.load == "eager" and override.load != "eager":
                raise AttributeError("Field protodescriptor load cannot be loosened.")


@declarator
class AssignableFieldProtodescriptor(AssignableDeclarator, IdentifiableDeclarator, FieldProtodescriptor):  # noqa
    """Abstract base class for assignable field descriptors ('data' and 'link')."""
    required: bool = field(default=False)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("required", self.required, bool)


@declarator
class RelationProtodescriptor(FieldProtodescriptor):
    """Base descriptor for relation fields."""
    __handle__ = "relation"


@declarator
class ConstructorDeclarator[C](CallableDeclarator[Callable[[type, Any], Any]], Protodescriptor):  # noqa
    """Base class for construction method descriptors."""
    __handle__ = "constructor"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    def __override__(self, override: Declarator) -> None:
        return super().__override__(override)


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

    def __validate__(self) -> None:
        super().__validate__()
        if not self.members:
            raise ValueError("AssignableFieldEnumeration must have at least one member.")

# endregion

# =============================================================================
# Concrete Protodescriptor classes
# =============================================================================
# region Concrete Protodescriptor classes


@declarator
class DataProtodescriptor(AssignableFieldProtodescriptor):
    """The protodescriptor class for data fields."""
    __handle__ = "data"
    index: bool = field(default=False)
    typekey: bool = field(default=False)
    key: bool = field(default=False)
    sequence: bool = field(default=False)
    timestamp: bool = field(default=False)
    start_time: bool = field(default=False)
    end_time: bool = field(default=False)
    period: bool = field(default=False)
    location: bool = field(default=False)
    geom: bool = field(default=False)
    gt: Real | Missing = field(default=MISSING)
    ge: Real | Missing = field(default=MISSING)
    lt: Real | Missing = field(default=MISSING)
    le: Real | Missing = field(default=MISSING)
    min_length: int | Missing = field(default=MISSING)
    max_length: int | Missing = field(default=MISSING)
    pattern: str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        for attr in ("index", "typekey", "key", "sequence", "timestamp",
                     "start_time", "end_time", "period", "location", "geom"):
            self._validate_value_type(attr, getattr(self, attr), bool)
        for attr in ("gt", "ge", "lt", "le"):
            if is_not_missing(value := getattr(self, attr)):
                self._validate_value_type(attr, value, Real)
        if is_not_missing(self.min_length):
            self._validate_value_type("min_length", self.min_length, int)
        if is_not_missing(self.max_length):
            self._validate_value_type("max_length", self.max_length, int)
        if is_not_missing(self.pattern):
            self._validate_value_type("pattern", self.pattern, str)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind a classvar data value, disallowing reassignment in subclasses."""
        if is_not_missing(previous) and value != previous:
            raise AttributeError(f"Data field '{self.name}' cannot be reassigned.")
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            raise ValueError(
                f"Inline validation failed for data '{self.name}' with value: {value}"
            )
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten data constraints only."""
        if type(override) is not type(self):
            raise AttributeError("Data protodescriptors can only be overridden by the same class.")
        if override == self:
            return
        self._validate_override_common(override)
        for attr in ("index", "required", "typekey", "key", "sequence", "timestamp",
                     "start_time", "end_time", "period", "location", "geom", "unique"):
            if not _tighten_bool(getattr(self, attr), getattr(override, attr)):
                raise AttributeError(f"Data protodescriptor '{attr}' cannot be loosened.")
        if not _tighten_lower(self.gt, self.ge, override.gt, override.ge):
            raise AttributeError("Data protodescriptor lower bound cannot be loosened.")
        if not _tighten_upper(self.lt, self.le, override.lt, override.le):
            raise AttributeError("Data protodescriptor upper bound cannot be loosened.")
        if is_not_missing(self.min_length) and is_not_missing(override.min_length):
            if override.min_length < self.min_length:
                raise AttributeError("Data protodescriptor min_length cannot be loosened.")
        if is_not_missing(self.max_length) and is_not_missing(override.max_length):
            if override.max_length > self.max_length:
                raise AttributeError("Data protodescriptor max_length cannot be loosened.")
        if is_not_missing(self.pattern) and is_not_missing(override.pattern):
            if override.pattern != self.pattern:
                raise AttributeError("Data protodescriptor pattern cannot be overridden.")

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate a data binding in the context of a host type."""
        if not self.classvar:
            raise AttributeError("Only data descriptors typed as class variables can be bound.")
        if is_missing(value):
            if self.default is not MISSING:
                return True
            else:
                raise ValueError("Datad binding cannot be missing.")
        elif value is None:
            if not self.nullable:
                raise ValueError("Non-nullable data field cannot be bound to None.")
        elif self.type is not None:
            if not isinstance(value, self.type):
                if not (isinstance(value, type) and issubclass(value, self.type)):
                    raise TypeError("Cannot bind data field of incompatible type.")
        return True


@declarator
class LinkProtodescriptor(AssignableFieldProtodescriptor, RelationProtodescriptor):
    """The protodescriptor class for link fields."""
    __handle__ = "link"
    on_delete: Literal["restrict", "set_null", "cascade"] = field(default="restrict")
    key: bool = field(default=False)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("on_delete", self.on_delete, str)
        self._validate_value_type("key", self.key, bool)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        raise AttributeError("Link fields are not prototype-bindable.")

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten link constraints only."""
        if type(override) is not type(self):
            raise AttributeError("Link protodescriptors can only be overridden by the same class.")
        if override == self:
            return
        self._validate_override_common(override)
        for attr in ("required", "key", "unique"):
            if not _tighten_bool(getattr(self, attr), getattr(override, attr)):
                raise AttributeError(f"Link protodescriptor '{attr}' cannot be loosened.")
        order = {"cascade": 0, "set_null": 1, "restrict": 2}
        if order.get(override.on_delete, 0) < order.get(self.on_delete, 0):
            raise AttributeError("Link protodescriptor on_delete cannot be loosened.")


@declarator
class BackLinkProtodescriptor(IdentifiableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for backlink fields."""
    __handle__ = "backlink"
    via: type | Missing = field(default=MISSING)
    limit: int | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.via):
            self._validate_value_type("via", self.via, type)
        if is_not_missing(self.limit):
            self._validate_value_type("limit", self.limit, int)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        raise AttributeError("Backlink fields are not prototype-bindable.")

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten backlink constraints only."""
        if type(override) is not type(self):
            raise AttributeError(
                "Backlink protodescriptors can only be overridden by the same class."
            )
        if override == self:
            return
        self._validate_override_common(override)
        if not _tighten_bool(self.unique, override.unique):
            raise AttributeError("Backlink protodescriptor 'unique' cannot be loosened.")
        if is_not_missing(self.via) and is_not_missing(override.via):
            if override.via != self.via:
                raise AttributeError("Backlink protodescriptor 'via' cannot be overridden.")
        if is_not_missing(self.limit) and is_not_missing(override.limit):
            if override.limit > self.limit:
                raise AttributeError("Backlink protodescriptor limit cannot be loosened.")


@declarator
class SelectionProtodescriptor(RelationProtodescriptor, CallableDeclarator[Callable]):
    """Protodescriptor for selection relations."""
    __handle__ = "selection"
    kind: Literal["sql", "ibis"] | Missing = field(default=MISSING)
    # Syntactic sugar for callable-based forms
    sql: Callable | Missing = field(default=MISSING)
    ibis: Callable | Missing = field(default=MISSING)
    # Field-expression-based form
    fx: Callable | str | Missing = field(default=MISSING)
    # Frame-based form
    source: Any | Missing = field(default=MISSING)
    key: Any | list[Any] | Missing = field(default=MISSING)
    time: Any | Missing = field(default=MISSING)
    space: Any | Missing = field(default=MISSING)
    filter: Callable | Any | Missing = field(default=MISSING)
    sort: str | list[str] | Missing = field(default=MISSING)
    limit: int | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()

        # Step 1: normalize callable forms
        callable_sources = {
            "callable": self.callable,
            "sql": self.sql,
            "ibis": self.ibis,
        }
        present_callables = {
            name: value
            for name, value in callable_sources.items()
            if is_not_missing(value)
        }

        if len(present_callables) > 1:
            raise TypeError(
                "SelectionProtodescriptor: callable, sql, and ibis are mutually exclusive."
            )

        if "sql" in present_callables:
            object.__setattr__(self, "callable", self.sql)
            object.__setattr__(self, "kind", "sql")
            object.__setattr__(self, "sql", MISSING)

        if "ibis" in present_callables:
            object.__setattr__(self, "callable", self.ibis)
            object.__setattr__(self, "kind", "ibis")
            object.__setattr__(self, "ibis", MISSING)

        if is_not_missing(self.callable):
            self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]
            if is_not_missing(self.kind):
                self._validate_value_type("kind", self.kind, str)
        else:
            if is_not_missing(self.kind):
                raise TypeError(
                    "SelectionProtodescriptor.kind is only valid when a callable is supplied."
                )

        # Step 2: detect which selection form is used
        has_callable = is_not_missing(self.callable)
        has_fx = is_not_missing(self.fx)
        has_frame = any(
            is_not_missing(v)
            for v in (
                self.source,
                self.key,
                self.time,
                self.space,
                self.filter,
                self.sort,
                self.limit,
            )
        )
        forms_used = sum((has_callable, has_fx, has_frame))
        if forms_used != 1:
            raise TypeError(
                "SelectionProtodescriptor must declare exactly one selection form: "
                "callable-based, fx-based, or frame-based."
            )

        # Step 3: validate per-form constraints
        if has_fx:
            self._validate_value_type("fx", self.fx, (Callable, str))  # type: ignore[arg-type]
        if has_frame:
            if is_not_missing(self.sort):
                if isinstance(self.sort, list):
                    if any(not isinstance(s, str) for s in self.sort):
                        raise TypeError("SelectionProtodescriptor.sort must be strings.")
                else:
                    self._validate_value_type("sort", self.sort, str)
            if is_not_missing(self.limit):
                self._validate_value_type("limit", self.limit, int)


@declarator
class DocumentProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for document fields."""
    __handle__ = "document"
    path: str | Missing = field(default=MISSING)
    format: str | Missing = field(default=MISSING)
    compression: str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.path):
            self._validate_value_type("path", self.path, str)
        if is_not_missing(self.format):
            self._validate_value_type("format", self.format, str)
        if is_not_missing(self.compression):
            self._validate_value_type("compression", self.compression, str)


@declarator
class FolderProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for folder fields."""
    __handle__ = "folder"
    path: str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.path):
            self._validate_value_type("path", self.path, str)


@declarator
class StateProtodescriptor(RelationProtodescriptor):
    """Protodescriptor for dynamic state values."""
    __handle__ = "state"
    # Underlying time series
    source: declarative | Protodescriptor | Missing = field(default=MISSING)
    # Reduction policy
    reducer: str | Callable | Missing = field(default=MISSING)
    # Constraints
    max_lag: str | timedelta | Missing = field(default=MISSING)
    min_observations: int | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.reducer):
            self._validate_value_type("reducer", self.reducer, (str, Callable))  # type: ignore[arg-type]
        if is_not_missing(self.max_lag):
            self._validate_value_type("max_lag", self.max_lag, (str, timedelta))  # type: ignore[arg-type]
        if is_not_missing(self.min_observations):
            self._validate_value_type("min_observations", self.min_observations, int)
            if self.min_observations < 1:
                raise ValueError("state.min_observations must be >= 1.")


@declarator
class FieldExpressionProtodescriptor(FieldProtodescriptor, CallableDeclarator[Callable[[declarative], Any]]):  # noqa
    """The protodescriptor class for field expressions."""
    __handle__ = "fx"
    ref: str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_missing(self.callable) and is_missing(self.ref):
            raise TypeError("FieldExpressionProtodescriptor requires either a 'callable' or 'ref' attribute.")  # noqa
        if is_not_missing(self.ref):
            self._validate_value_type("ref", self.ref, str)
        if is_not_missing(self.callable):
            self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class FieldGroupProtodescriptor(FieldProtodescriptor, FieldEnumeration):
    """The protodescriptor class for field groups."""
    __handle__ = "fieldgroup"

    def __validate__(self) -> None:
        super().__validate__()
        if not self.members:
            raise ValueError("FieldGroupProtodescriptor must have at least one member.")


@declarator
class FieldBlockProtodescriptor(FieldProtodescriptor):
    """The protodescriptor class for field blocks."""
    __handle__ = "fieldblock"


@declarator
class MetricProtodescriptor(FieldProtodescriptor, CallableDeclarator[Callable[[type], Any]]):  # noqa
    """The protodescriptor class for metric fields."""
    __handle__ = "metric"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class ParserDeclarator(ConstructorDeclarator[Callable[[type, Any], Any]], FieldEnumeration):  # noqa
    """The declarator class for field parsers."""
    __handle__ = "parser"
    element_wise: bool = field(default=False)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("element_wise", self.element_wise, bool)


@declarator
class ValidatorDeclarator(ConstructorDeclarator[Callable[[type, Any], bool]], FieldEnumeration):  # noqa
    """The declarator class for field validators."""
    __handle__ = "validator"
    element_wise: bool = field(default=False)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("element_wise", self.element_wise, bool)


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
    kind: str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.kind):
            self._validate_value_type("kind", self.kind, str)


@declarator
class PartitionDeclarator(SchemaDeclarator):
    """The declarator class for schema partitions."""
    __handle__ = "partition"
    key: FieldProtodescriptor = field()
    scheme: Literal["range", "categorical", "hash", "time"] = field()
    # MISSING → not specified (inherit / infer)
    # None → scheme-defined implicit bucketing (categorical, hash)
    buckets: None | int | tuple | list | str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("key", self.key, FieldProtodescriptor)
        self._validate_value_type("scheme", self.scheme, str)
        if is_not_missing(self.buckets):
            self._validate_value_type("buckets", self.buckets, (type(None), int, tuple, list, str))  # type: ignore[arg-type]


@declarator
class PathDeclarator(SchemaDeclarator):
    """The declarator class for traversal paths."""
    __handle__ = "path"
    template: str | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.template):
            self._validate_value_type("template", self.template, str)


@declarator
class SortDeclarator(FieldEnumeration, SchemaDeclarator):
    """The declarator class for schema sort orders."""
    __handle__ = "sort"
    sort_directions: Literal["asc", "desc"] | list[Literal["asc", "desc"]] | Missing = field(default=MISSING)  # noqa

    def __validate__(self) -> None:
        super().__validate__()
        if not self.members:
            raise ValueError("SortDeclarator must have at least one member.")
        if is_not_missing(self.sort_directions):
            if not isinstance(self.sort_directions, (str, list)):
                raise TypeError("SortDeclarator.sort_directions must be 'asc' or 'desc' or a list thereof.")  # noqa
            if isinstance(self.sort_directions, str):
                if self.sort_directions not in {"asc", "desc"}:
                    raise ValueError("SortDeclarator.sort_directions must be 'asc' or 'desc'.")
            else:
                invalid = [v for v in self.sort_directions if v not in {"asc", "desc"}]
                if invalid:
                    raise TypeError("SortDeclarator.sort_directions must be 'asc' or 'desc'.")
                if not len(self.sort_directions) == len(self.members):
                    raise ValueError("SortDeclarator.sort_directions length must match members length.")  # noqa

# endregion

# =============================================================================
# Registry class
# =============================================================================
# region Registry class


class ProtodescriptorRegistry(MultiRegistry):
    """Registry for meta bindings and protodescriptors."""

    __handles__ = {"metadata", "nxfield", "option", "field", "schema", "constructor", "data"}

    def __init__(self, metadescriptors: MetadescriptorRegistry):
        metadata = BindingRegistry(_declarators=metadescriptors.metadata)
        nxfield = BindingRegistry(_declarators=metadescriptors.nxfield)
        option = BindingRegistry(_declarators=metadescriptors.option)
        field = DeclaratorRegistry[FieldProtodescriptor]()
        schema = DeclaratorRegistry[SchemaDeclarator]()
        constructor = DeclaratorRegistry[ConstructorDeclarator]()
        data = BindingRegistry(_declarators=field)
        super().__init__(
            metadata=metadata,
            nxfield=nxfield,
            option=option,
            field=field,
            schema=schema,
            constructor=constructor,
            data=data,
        )
        self._metadescriptors = metadescriptors

    @property
    def metadescriptors(self) -> MetadescriptorRegistry | None:
        """Returns the metadescriptor registry, or None if it has been garbage collected."""
        return self._metadescriptors

    def __copy__(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        new_registry = self.__class__(self._metadescriptors.copy())
        new_registry.update(self)
        return new_registry

    @property
    def metadata(self) -> BindingRegistry[MetadataDeclarator]:
        """Returns the metadata binding registry."""
        return cast(BindingRegistry[MetadataDeclarator], self._registries["metadata"])

    @property
    def nxfield(self) -> BindingRegistry[NxFieldDeclarator]:
        """Returns the nxfield binding registry."""
        return cast(BindingRegistry[NxFieldDeclarator], self._registries["nxfield"])

    @property
    def option(self) -> BindingRegistry[OptionDeclarator]:
        """Returns the option binding registry."""
        return cast(BindingRegistry[OptionDeclarator], self._registries["option"])

    @property
    def field(self) -> DeclaratorRegistry[FieldProtodescriptor]:
        """Returns the field protodescriptor registry."""
        return cast(DeclaratorRegistry[FieldProtodescriptor], self._registries["field"])

    @property
    def schema(self) -> DeclaratorRegistry[SchemaDeclarator]:
        """Returns the schema declarator registry."""
        return cast(DeclaratorRegistry[SchemaDeclarator], self._registries["schema"])

    @property
    def constructor(self) -> DeclaratorRegistry[ConstructorDeclarator]:
        """Returns the constructor declarator registry."""
        return cast(DeclaratorRegistry[ConstructorDeclarator], self._registries["constructor"])

    @property
    def data(self) -> BindingRegistry[DataProtodescriptor]:
        """Returns the data binding registry."""
        return cast(BindingRegistry[DataProtodescriptor], self._registries["data"])

    @classmethod
    def handle(cls, declarator: Declarator) -> str | None:
        """Returns the registry handle for the given declarator, or None if not found."""
        match declarator:
            case FieldProtodescriptor():
                return "field"
            case SchemaDeclarator():
                return "schema"
            case ConstructorDeclarator():
                return "constructor"
            case _:
                return None

    def register(self, name: str, item: Any, *, handle: str | None = None) -> None:
        """Registers a declarator or binding in the appropriate registry."""
        if isinstance(item, Declarator):
            return super().register(name, item, handle=handle)
        # If the item is a binding, try to register in each binding registry
        for binding_registry in (self.metadata, self.nxfield, self.option, self.data):
            try:
                binding_registry.register(name, item)
                return
            except KeyError:
                continue
        raise KeyError(f"Cannot register item {item} with name {name} in any registry.")

# endregion
