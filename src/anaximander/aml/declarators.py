# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Define declarators, protodescriptors, and metadescriptors for AML.

Declarators are the core semantic building blocks of AML: they capture the
metadata, field, and validation contracts that declarative class bodies express.
This module defines the base declarator hierarchy, the metadescriptors that live
under the ``nx`` inner interface, and the protodescriptors that materialize
instance-level fields such as data and links.

The declarator z-ordering encoded here drives manifest serialization and the
ordering of handle signatures. It therefore acts as a source of truth for the
user-facing DSL surface and for downstream compiler expectations.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import ast
import builtins
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import timedelta
from itertools import chain
from numbers import Real
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    dataclass_transform,
)

import attrs
import yaml

from ..utils.meta import classproperty
from ..utils.yaml import (
    nx_register_constructor,
    nx_register_multi_representer,
    nx_register_representer,
    nx_register_type,
    nx_set_ignore_aliases,
    nx_yaml_dump,
)
from .basetypes import is_metadata_type
from .diagnostics import (
    DECLARATOR_DIAGNOSTICS,
    DeclaratorDiagnosticBag,
    DeclaratorHandleConflict,
    DeclaratorHandleInvalid,
    DeclaratorInvariantViolation,
    DeclaratorNullabilityViolation,
    DeclaratorOverrideViolation,
    DeclaratorReassignmentViolation,
    DeclaratorTypeConstraintViolation,
)

# endregion

# =============================================================================
# Helpers and globals
# =============================================================================
# region Constants

# Helpers for Declarator classes
DT = TypeVar("DT", bound=type)  # Declarator-type type variable
T = TypeVar("T")


def field(*args, z: int | None = None, **kwargs):
    """Declare a field with optional z-order metadata.

    Args:
        *args: Positional arguments forwarded to attrs.field.
        z: Optional z-order metadata.
        **kwargs: Keyword arguments forwarded to attrs.field.

    Returns:
        The configured attrs field.
    """
    metadata = dict(kwargs.pop("metadata", {}))
    if z is not None:
        metadata["z"] = z
    return attrs.field(*args, metadata=metadata, **kwargs)


@dataclass_transform(kw_only_default=True, field_specifiers=(field,))
def declarator(cls: DT) -> DT:
    """Apply attrs define for declarator classes with a preserved init signature.

    Args:
        cls: Declarator class to decorate.

    Returns:
        The decorated declarator class.
    """
    return attrs.define(cls, slots=False, frozen=True, kw_only=True, repr=False, str=False)


# Missing sentinel
class _MissingSentinel:
    """Sentinel object representing the absence of a declarative value.

    This is distinct from None, False, 0, or empty containers.
    It is used to indicate that an attribute was not declared.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the sentinel representation."""
        return "MISSING"

    def __bool__(self) -> bool:
        """Disallow truthiness for the sentinel."""
        DeclaratorInvariantViolation.report("MISSING has no truth value")
        return False

    def __eq__(self, other: object) -> bool:
        """Return identity equality for the sentinel."""
        return self is other

    def __ne__(self, other: object) -> bool:
        """Return identity inequality for the sentinel."""
        return self is not other

    def __hash__(self) -> int:
        """Return a stable hash for the sentinel."""
        return id(self)

    def __reduce__(self):
        """Return reducer that preserves singleton behavior."""
        # Ensures singleton behavior across pickling.
        return "MISSING"

Missing: TypeAlias = _MissingSentinel

# The one and only instance
MISSING = _MissingSentinel()


def _register_aml_yaml_declarators() -> None:
    """Register YAML handlers for declarators and missing sentinel."""
    global register_yaml_type

    def register_yaml_type(name: str, type_: type) -> None:
        """Register a custom type name for YAML resolution."""
        nx_register_type(name, type_)

    def _repr_declarator(dumper, obj: "Declarator"):
        """Represent declarators as scalar references."""
        return dumper.represent_scalar("tag:yaml.org,2002:str", repr(obj))

    def _repr_missing(dumper, obj: Missing):
        """Represent the missing sentinel as a scalar."""
        return dumper.represent_scalar("tag:yaml.org,2002:str", "MISSING")

    def _repr_mappingproxy(dumper, obj: MappingProxyType):
        """Represent mapping proxy values as standard mappings."""
        return dumper.represent_mapping("tag:yaml.org,2002:map", dict(obj))

    def _construct_missing(loader, node):
        """Construct the missing sentinel from a YAML node."""
        loader.construct_scalar(cast(yaml.ScalarNode, node))
        return MISSING

    nx_set_ignore_aliases(lambda value: isinstance(value, Missing))
    nx_register_representer(Declarator, _repr_declarator)
    nx_register_multi_representer(Declarator, _repr_declarator)
    nx_register_representer(Missing, _repr_missing)
    nx_register_representer(MappingProxyType, _repr_mappingproxy)
    nx_register_constructor("!Missing", _construct_missing)


def is_missing(value: object) -> TypeGuard[Missing]:
    """Return whether a value is the MISSING sentinel."""
    return value is MISSING


def is_not_missing(value: T | Missing) -> TypeGuard[T]:
    """Return whether a value is not the MISSING sentinel."""
    return value is not MISSING


# Helper functions for declarator hooks
def _tighten_type(base: type | None, override: type | None) -> bool:
    """Return whether a type override is a monotone tightening."""
    if base is None or override is None:
        return True
    try:
        return issubclass(override, base)
    except TypeError:
        return False


def _tighten_nullable(base: bool | None, override: bool | None) -> bool:
    """Return whether a nullable override is monotone tightening."""
    if base is None or override is None:
        return True
    return not (base is False and override is True)


def _tighten_classvar(base: bool | None, override: bool | None) -> bool:
    """Return whether a classvar override preserves classvar semantics."""
    if base is None or override is None:
        return True
    return base == override


def _tighten_bool(base: bool, override: bool) -> bool:
    """Return whether a boolean override is monotone tightening."""
    return not base or override


def _tighten_bound_value(value: Any, previous: Any) -> bool:
    """Return whether a bound value is a monotone tightening of its predecessor."""
    if isinstance(previous, type) and isinstance(value, type):
        return _tighten_type(previous, value)
    return value == previous


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
    """Return whether the lower-bound constraint is monotonically tightened."""
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
    """Return whether the upper-bound constraint is monotonically tightened."""
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

# endregion

# =============================================================================
# Declarator base class
# =============================================================================
# region Declarator base class


@attrs.define(frozen=True, order=True)
class DeclarativeTypeKey:
    """A unique key for identifying declarative types."""
    project: str
    module: str
    name: str


class DeclarativeProtocol(Protocol):
    """A protocol for declarative types with unique keys."""
    __key__: ClassVar[DeclarativeTypeKey]

Declarative = type[DeclarativeProtocol]


@attrs.define(frozen=True, order=True)
class DeclaratorKey:
    """A unique key for identifying declarators."""
    owner: DeclarativeTypeKey
    ordinal: int


class DeclaratorConfig(Protocol):
    """A protocol for extending declarator configuration."""

    def resolve(self, **context) -> Mapping[str, Any]:
        """Resolve configuration values against a context mapping.

        Args:
            **context: Context values used to resolve configuration entries.

        Returns:
            A mapping of resolved configuration values.
        """
        ...


type ConfigValue = Any | DeclaratorConfig | Mapping[str, ConfigValue]
type Config = DeclaratorConfig | Mapping[str, ConfigValue]


@declarator
class Declarator(ABC):
    """Base class for all declarators.

    Declarators are used to declare named attributes or add features to classes that use them,
    working in conjunction with the declarative metaclass to process and register these
    declarations.
    """

    # Reserved names that cannot be used for declarators.
    # These can be either strings or compiled regex patterns.
    # Strings restrict exact matches, while regex patterns allow for more complex rules.
    __reserved_patterns__: ClassVar[set[str | re.Pattern[str]]] = {
        re.compile(r"^__.*"),
        re.compile(r".*\..*"),
    }

    __handles__: ClassVar[dict[str, type]] = {}  # Mapping of handles to declarator types (do not override) # noqa
    __handle__: ClassVar[str] = ""  # A plain-text lowercase handle for this declarator type (override in subclasses) # noqa
    _handles: ClassVar[tuple[str, ...]] = ()  # A tuple of all handles for this declarator type and its ancestors, in reverse mro (do not override) # noqa

    # Post-init wired fields (logically immutable; set via internal backdoor).
    name: str = field(init=False, default=None, z=100)  # Attribute or key name this declarator is assigned to # noqa
    owner: Declarative = field(init=False, default=None, z=110)  # Owning class of this declarator
    ordinal: int = field(init=False, default=None, z=120)  # Index of this declarator within the owning class # noqa

    # Init-time fields (immutable)
    doc: str | Missing = field(default=MISSING, z=900)  # Optional documentation string # noqa
    config: Mapping[str, ConfigValue] | Missing = field(factory=dict, z=910)  # Extraneous declarator configuration # noqa

    @property
    def __key__(self) -> DeclaratorKey:
        """A unique key for identifying this declarator."""
        if self.owner is None or self.ordinal is None:
            DeclaratorInvariantViolation.report(
                "Declarator must be bound to a class before accessing its key."
            )
            return DeclaratorKey(owner=DeclarativeTypeKey("anaximander", "unknown", "unknown"), ordinal=-1)  # noqa: E501
        return DeclaratorKey(
            owner=self.owner.__key__,
            ordinal=self.ordinal,
        )

    @classproperty
    def handle(cls) -> str:
        """The plain-text lowercase handle for this declarator type."""
        return cls.__handle__

    @classproperty
    def handles(cls) -> tuple[str, ...]:
        """A tuple of all handles for this declarator type and its ancestors."""
        return cls._handles

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the declarator's type."""
        if cls.__handle__:
            return f"{cls.__handle__} declarator"
        return cls.__name__

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values in the domain namespace."""
        return False

    def __attrs_post_init__(self) -> None:
        """Post-initialization processing for the declarator."""
        # Attach diagnostics immediately for IDE-visible availability.
        if not hasattr(self, "__diagnostics__"):
            object.__setattr__(self, "__diagnostics__", DeclaratorDiagnosticBag(self))
        if not hasattr(self, "__ast__"):
            object.__setattr__(self, "__ast__", None)
        # Freeze config to prevent accidental mutation.
        if isinstance(self.config, Mapping):
            object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    def __init_subclass__(cls):
        super().__init_subclass__()
        base_declarators = (b for b in cls.__bases__ if issubclass(b, Declarator))
        # Resolve reserved patterns from base classes
        base_reserved_patterns = (b.__reserved_patterns__ for b in base_declarators)
        reserved_patterns = set(chain.from_iterable(base_reserved_patterns))
        if "__reserved_patterns__" in vars(cls):
            try:
                assert all(
                    isinstance(pattern, (str, re.Pattern)) for pattern in cls.__reserved_patterns__
                )
            except AssertionError:
                DeclaratorTypeConstraintViolation.report(
                    "All elements of __reserved_patterns__ must be instances of str or re.Pattern."
                )
                return
            cls.__reserved_patterns__ = reserved_patterns | set(cls.__reserved_patterns__)
        else:
            cls.__reserved_patterns__ = reserved_patterns
        # Register and resolve handles
        if cls.__handle__ == "" and any(b.__handle__ != "" for b in base_declarators):  # noqa
            DeclaratorHandleInvalid.report(
                "Declarator subclasses cannot define an empty __handle__ if any base class does not."
            )
            return
        if cls.__handle__ != "":
            if cls.__handle__ in cls.__handles__:
                if cls.__handles__[cls.__handle__] not in cls.mro():
                    DeclaratorHandleConflict.report(
                        f"Declarator handle '{cls.__handle__}' is already registered."
                    )
                    return
            cls.__handles__[cls.__handle__] = cls
        handles = []
        for base in reversed(cls.mro()):
            if issubclass(base, Declarator) and base.__handle__ != "":
                handles.append(base.__handle__)
        cls._handles = tuple(handles)

    def _set_once(self, attr: str, value: Any, *, treat_none_as_unset: bool = False) -> None:
        """Internal backdoor: set a frozen attribute once (or idempotently)."""
        current = getattr(self, attr)
        if is_missing(current) or (treat_none_as_unset and current is None):
            object.__setattr__(self, attr, value)
            return
        if current != value:
            DeclaratorInvariantViolation.report(
                f"{self.__class__.__name__}.{attr} is already set."
            )
            return
        # idempotent re-set
        object.__setattr__(self, attr, value)

    def _validate_value_type(
        self,
        attr: str,
        value: Any,
        expected: type | tuple[type, ...],
    ) -> None:
        """Validate a value against its expected runtime type."""
        if is_missing(value):
            DeclaratorInvariantViolation.report(
                f"{self.__class__.__name__}.{attr} is missing."
            )
            return

        if not isinstance(value, expected):
            if isinstance(expected, tuple):
                expected_name = " | ".join(t.__name__ for t in expected)
            else:
                expected_name = expected.__name__
            DeclaratorTypeConstraintViolation.report(
                f"{self.__class__.__name__}.{attr} must be {expected_name}, "
                f"got {type(value).__name__}."
            )
            return

    def __set_name__(self, owner: type, name: str):
        """Attach the name and owner class to this declarator."""
        token = DECLARATOR_DIAGNOSTICS.set(self.__diagnostics__)
        self._validate_value_type("name", name, str)
        self._validate_value_type("owner", owner, type)
        forbidden_names = {
            pattern for pattern in self.__reserved_patterns__ if isinstance(pattern, str)
        }
        forbidden_patterns = {
            pattern for pattern in self.__reserved_patterns__ if isinstance(pattern, re.Pattern)
        }
        if name in forbidden_names:
            DeclaratorTypeConstraintViolation.report(
                f"Cannot use reserved name {name} for {self.typename}."
            )
            return
        if any(pattern.fullmatch(name) for pattern in forbidden_patterns):
            DeclaratorTypeConstraintViolation.report(
                f"Cannot use reserved name {name} for {self.typename}."
            )
            return
        self._set_once("name", name, treat_none_as_unset=True)
        self._set_once("owner", owner, treat_none_as_unset=True)
        DECLARATOR_DIAGNOSTICS.reset(token)

    def __set_ast__(self, node: ast.AST | None) -> None:
        """Attach the AST node that declared this declarator (if any)."""
        self._validate_value_type("__ast__", node, (ast.AST, type(None)))
        object.__setattr__(self, "__ast__", node)

    def __validate__(self) -> None:
        """Validate this declarator after it has been bound to a class namespace.

        This method is called by the declarative metaclass after the declarator
        has been assigned to a class attribute, but before the class is finalized.
        Its role is to validate the declarator's attributes and configuration,
        absent any context. Subclasses can override this method to implement custom
        validation logic.
        """
        if is_not_missing(self.doc):
            self._validate_value_type("doc", self.doc, (str, type(None)))
        if is_not_missing(self.config):
            self._validate_value_type("config", self.config, (Mapping, type(None)))
        self._validate_value_type("name", self.name, str)
        self._validate_value_type("owner", self.owner, type)
        self._validate_value_type("ordinal", self.ordinal, int)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Hook called when this declarator is bound to a value.

        This method can be overridden by subclasses to customize the binding behavior.
        The default implementation raises an AttributeError.
        """
        DeclaratorOverrideViolation.report(
            f"Declarator of type {self.__class__.__name__} cannot be bound to a value."
        )
        return previous

    @abstractmethod
    def __override__(self, override: "Declarator") -> None:
        """Hook called when this declarator is overridden in a subclass.

        This method can be overridden by subclasses to customize the override behavior.
        The default implementation raises an AttributeError.
        """
        if override == self:
            return
        DeclaratorOverrideViolation.report(
            f"Declarator of type {self.__class__.__name__} cannot be overridden."
        )
        return

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Hook called to validate a bound value in the context of a hosting class.

        Unlike __bind__, which is called when the binding occurs and does not use context,
        this method is designed to be called at AML module finalization.
        This method can be overridden by subclasses to implement custom validation logic.
        The default implementation returns True.
        """
        return True

    def __repr__(self) -> str:
        if self.owner is not None:
            if self.name is not None:
                return f"<{self.owner.__name__}.{self.name} {self.typename}>"
            return f"<{self.owner.__name__} {self.typename}>"
        return f"<unbound {self.typename}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict suitable for serialization."""
        attributes = attrs.fields(self.__class__)
        ordered = []
        for idx, attr in enumerate(attributes):
            z = attr.metadata.get("z", 1000)
            if not isinstance(z, int):
                z = 1000
            ordered.append((z, idx, attr.name))
        ordered.sort()
        return {name: getattr(self, name) for _, _, name in ordered}

    def to_yaml(self) -> str:
        """YAML-style pretty print."""
        return nx_yaml_dump(self.to_dict())

    def __str__(self) -> str:
        return self.to_yaml()


_register_aml_yaml_declarators()

@declarator
class AnnotatableDeclarator(Declarator):
    """Base class for declarators that can be annotated with type information."""
    annotation: str | None = field(init=False, default=None, z=200)  # Literal type annotation as a string  # noqa
    hint: Any | None = field(init=False, default=None, z=210)  # Evaluated type hint object  # noqa
    type: builtins.type | None = field(init=False, default=None, z=220)  # Evaluated type annotation  # noqa
    nullable: bool | None = field(init=False, default=None, z=230)  # Whether the type is nullable  # noqa
    classvar: bool | None = field(init=False, default=None, z=240)  # Whether the type is a ClassVar  # noqa
    def __init_subclass__(cls):
        super().__init_subclass__()

    def __validate_annotation__(self, annotation: str | None, hint: Any | None) -> None:
        """Validate a resolved annotation for this declarator.

        This hook focuses on annotation semantics, not runtime binding checks.
        Subclasses should validate the effective hint and ignore missing hints.
        """
        return None

    def __set_type__(
        self,
        annotation: str | None,
        type_: Any | None,
        nullable: bool | None,
        classvar: bool | None = None,
        hint: Any | None = None,
    ):
        """Sets the type by supplying annotation, evaluated type, nullability and classvar."""
        self._validate_value_type("annotation", annotation, (str, type(None)))
        self._validate_value_type("type", type_, (type, type(None)))
        self._validate_value_type("nullable", nullable, (bool, type(None)))
        self._validate_value_type("classvar", classvar, (bool, type(None)))
        # Annotation validation is declarator-specific and may rely on the hint.
        self.__validate_annotation__(annotation, hint)
        self._set_once("annotation", annotation, treat_none_as_unset=True)
        self._set_once("hint", hint, treat_none_as_unset=True)
        self._set_once("type", type_, treat_none_as_unset=True)
        self._set_once("nullable", nullable, treat_none_as_unset=True)
        self._set_once("classvar", classvar, treat_none_as_unset=True)


@declarator
class IdentifiableDeclarator(AnnotatableDeclarator):
    """Base class for declarators of attributes that can uniquely identify an instance."""
    unique: bool = field(default=False, z=325)  # Whether this declarator uniquely identifies an instance  # noqa

    def __validate__(self) -> None:
        self._validate_value_type("unique", self.unique, bool)
        return super().__validate__()


@declarator
class AssignableDeclarator(AnnotatableDeclarator):
    """Base class for declarators of attributes that receive their value through assignment."""
    default: Any = field(default=MISSING, z=250)
    factory: Callable[[], Any] | Missing = field(default=MISSING, z=260)
    # This is a fail-quick optional inline validator that takes a value as its only argument
    # It is intended to be called in the __bind__ hook to validate assigned values
    validator: Callable[[Any], bool] | Missing = field(default=MISSING, z=520)

    def __validate__(self) -> None:
        if is_not_missing(self.factory):
            self._validate_value_type("factory", self.factory, Callable)  # type: ignore[arg-type]
        if is_not_missing(self.validator):
            self._validate_value_type("validator", self.validator, Callable)  # type: ignore[arg-type]
        return super().__validate__()


@declarator
class CallableDeclarator[C: Callable](Declarator):
    """A mixin class for declarators that wrap callables."""
    callable: C | Missing = field(default=MISSING, z=270)

    def __validate__(self) -> None:
        if is_not_missing(self.callable):
            self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]
        return super().__validate__()


@declarator
class EnumerationDeclarator(Declarator):
    """A mixin class for declarators that reference a list of declarators by name."""
    members: tuple[str, ...] = field(factory=tuple, z=280)
    __member_types__: ClassVar[tuple[type[Declarator], ...]] = ()  # Admissible member types

    def __validate__(self) -> None:
        self._validate_value_type("members", self.members, tuple)
        if any(not isinstance(member, str) for member in self.members):
            DeclaratorTypeConstraintViolation.report(
                "EnumerationDeclarator.members must be a tuple of strings."
            )
        return super().__validate__()

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Check that __member_types__ are tightening the admissible types from base classes.
        base_enumerations = (b for b in cls.__bases__ if issubclass(b, EnumerationDeclarator))
        base_mbtypes = tuple(chain.from_iterable(b.__member_types__ for b in base_enumerations))
        if "__member_types__" in vars(cls):
            mbtypes: tuple[type, ...] = cls.__member_types__
            try:
                assert all(issubclass(t, base_type) for t in mbtypes for base_type in base_mbtypes)
            except AssertionError:
                DeclaratorTypeConstraintViolation.report(
                    "__member_types__ must only contain types that are subclasses of all "
                    "admissible types from base classes."
                )
                return


@declarator
class EnumerationCallableDeclarator[C: Callable](CallableDeclarator[C], EnumerationDeclarator):  # noqa
    """A mixin class for callable declarators that reference a list of declarators by name."""
    pass

# endregion

# =============================================================================
# Metadescriptor classes
# =============================================================================
# region Metadescriptor classes


@declarator
class Metadescriptor(Declarator):
    """Base class for prototype-level descriptors declared in archetypes and traits."""
    __handle__ = "meta"
    __reserved_patterns__ = {re.compile(r"^nx.*")}


@declarator
class AssignableMetadescriptor(AssignableDeclarator, Metadescriptor):
    """Base class for assignable prototype-level descriptors declared in archetypes and traits."""


@declarator
class MetadataDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for metadata fields."""
    __handle__ = "metadata"
    domain: bool = field(default=False, z=305)  # Whether this metadata is part of the domain schema  # noqa

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values."""
        return self.domain

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("domain", self.domain, bool)

    def __validate_annotation__(self, annotation: str | None, hint: Any | None) -> None:
        """Validate metadata annotations against the metadata type contract."""
        if hint is not None and not is_metadata_type(hint):
            DeclaratorTypeConstraintViolation.report(
                f"Metadata '{self.name}' must be annotated as a metadata type."
            )
            return

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate a metadata binding in the context of a host type."""
        # Missing bindings are not validated here; only bound values are checked.
        if value is None:
            if not self.nullable:
                DeclaratorNullabilityViolation.report(
                    "Non-nullable metadata cannot be bound to None."
                )
        elif self.type is not None:
            if not isinstance(value, self.type):
                if not (isinstance(value, type) and issubclass(value, self.type)):
                    DeclaratorTypeConstraintViolation.report(
                        "Cannot bind metadata of incompatible type."
                    )
        return True

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind a metadata value, enforcing monotone tightening and inline validation."""
        if is_not_missing(previous) and value != previous:
            if not _tighten_bound_value(value, previous):
                DeclaratorReassignmentViolation.report(
                    f"Metadata '{self.name}' cannot be reassigned with a looser value."
                )
                return previous
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            DeclaratorTypeConstraintViolation.report(
                f"Inline validation failed for metadata '{self.name}' with value: {value}."
            )
            return previous
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten type constraints only."""
        if type(override) is not type(self):
            DeclaratorOverrideViolation.report(
                "Metadata declarators can only be overridden by the same class."
            )
            return
        if override == self:
            return
        if self.domain != override.domain:
            DeclaratorOverrideViolation.report(
                "Metadata declarator 'domain' cannot be overridden."
            )
            return
        if not _tighten_type(self.type, override.type):
            DeclaratorOverrideViolation.report("Metadata declarator type cannot be loosened.")
            return
        if not _tighten_nullable(self.nullable, override.nullable):
            DeclaratorOverrideViolation.report(
                "Metadata declarator nullability cannot be loosened."
            )
            return
        if not _tighten_classvar(self.classvar, override.classvar):
            DeclaratorOverrideViolation.report(
                "Metadata declarator classvar cannot be overridden."
            )
            return


@declarator
class OptionDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for option fields."""
    __handle__ = "option"

    def __validate_annotation__(self, annotation: str | None, hint: Any | None) -> None:
        """Validate option annotations against the metadata type contract."""
        if hint is not None and not is_metadata_type(hint):
            DeclaratorTypeConstraintViolation.report(
                f"Option '{self.name}' must be annotated as a metadata type."
            )
            return

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate an option binding in the context of a host type."""
        # Missing bindings are not validated here; only bound values are checked.
        if value is None:
            if not self.nullable:
                DeclaratorNullabilityViolation.report(
                    "Non-nullable option cannot be bound to None."
                )
        elif self.type is not None:
            if not isinstance(value, self.type):
                if not (isinstance(value, type) and issubclass(value, self.type)):
                    DeclaratorTypeConstraintViolation.report(
                        "Cannot bind option of incompatible type."
                    )
        return True


@declarator
class NxFieldDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"
    fieldtype: type = field(z=290)  # The expected type of the field reference

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("fieldtype", self.fieldtype, type)

    def __validate_annotation__(self, annotation: str | None, hint: Any | None) -> None:
        """Validate nxfield annotations as string references."""
        if hint is not None and hint is not str:
            DeclaratorTypeConstraintViolation.report(
                f"NxField '{self.name}' must be annotated as str."
            )
            return

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind an nxfield value, disallowing reassignment in derived prototypes."""
        if is_not_missing(previous) and value != previous:
            DeclaratorReassignmentViolation.report(
                f"NxField '{self.name}' cannot be reassigned."
            )
            return previous
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            DeclaratorTypeConstraintViolation.report(
                f"Inline validation failed for nxfield '{self.name}' with value: {value}."
            )
            return previous
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten fieldtype/type constraints only."""
        if type(override) is not type(self):
            DeclaratorOverrideViolation.report(
                "NxField declarators can only be overridden by the same class."
            )
            return
        if override == self:
            return
        if not _tighten_type(self.fieldtype, override.fieldtype):
            DeclaratorOverrideViolation.report(
                "NxField declarator fieldtype cannot be loosened."
            )
            return
        if not _tighten_type(self.type, override.type):
            DeclaratorOverrideViolation.report("NxField declarator type cannot be loosened.")
            return
        if not _tighten_nullable(self.nullable, override.nullable):
            DeclaratorOverrideViolation.report(
                "NxField declarator nullability cannot be loosened."
            )
            return
        if not _tighten_classvar(self.classvar, override.classvar):
            DeclaratorOverrideViolation.report(
                "NxField declarator classvar cannot be overridden."
            )
            return


# Metavalidators

@declarator
class MetaValidator(Metadescriptor):
    """A common base class for metadescriptor validators."""
    __handle__ = "metavalidator"

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the metavalidator's type."""
        return "metavalidator"


@declarator
class PrototypeValidator(CallableDeclarator[Callable[[type], bool]], MetaValidator):
    """The metadescriptor class for prototype validators."""
    __handle__ = "prototype_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the metavalidator's type."""
        return "prototype validator"


@declarator
class MetadataValidator(EnumerationCallableDeclarator[Callable[[type, Any], bool]], MetaValidator):  # noqa
    """The metadescriptor class for metadata validators."""
    __handle__ = "metadata_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the metavalidator's type."""
        return "metadata validator"


@declarator
class OptionValidator(EnumerationCallableDeclarator[Callable[[type, Any], bool]], MetaValidator):  # noqa
    """The metadescriptor class for option validators."""
    __handle__ = "option_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the metavalidator's type."""
        return "option validator"


@declarator
class NxFieldValidator(EnumerationCallableDeclarator[Callable[[type, Any], bool]], MetaValidator):  # noqa
    """The metadescriptor class for nxfield validators."""
    __handle__ = "nxfield_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the metavalidator's type."""
        return "nxfield validator"

# endregion

# =============================================================================
# Base Protodescriptors
# =============================================================================
# region Base Protodescriptors


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
    load: Literal["eager", "lazy"] | Missing = field(default=MISSING, z=320)
    repr: bool | Callable | str | Missing = field(default=MISSING, z=810)

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values."""
        return bool(self.classvar)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.load):
            if self.load not in {"eager", "lazy"}:
                DeclaratorTypeConstraintViolation.report(
                    "Field protodescriptor's load attribute must be 'eager' or 'lazy'."
                )
                return
        if is_not_missing(self.repr):
            self._validate_value_type("repr", self.repr, (bool, Callable, str))  # type: ignore[arg-type]

    def _validate_override_common(self, override: "FieldProtodescriptor") -> None:
        """Validate shared tightening rules for field protodescriptors."""
        if not _tighten_type(self.type, override.type):
            DeclaratorOverrideViolation.report("Field protodescriptor type cannot be loosened.")
            return
        if not _tighten_nullable(self.nullable, override.nullable):
            DeclaratorOverrideViolation.report(
                "Field protodescriptor nullability cannot be loosened."
            )
            return
        if not _tighten_classvar(self.classvar, override.classvar):
            DeclaratorOverrideViolation.report(
                "Field protodescriptor classvar cannot be overridden."
            )
            return
        if is_not_missing(self.load) and is_not_missing(override.load):
            if self.load == "eager" and override.load != "eager":
                DeclaratorOverrideViolation.report(
                    "Field protodescriptor load cannot be loosened."
                )
                return


@declarator
class AssignableFieldProtodescriptor(AssignableDeclarator, IdentifiableDeclarator, FieldProtodescriptor):  # noqa
    """Abstract base class for assignable field descriptors ('data' and 'link')."""
    required: bool = field(default=False, z=315)

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
            DeclaratorTypeConstraintViolation.report(
                "AssignableFieldEnumeration must have at least one member."
            )
            return

# endregion

# =============================================================================
# Concrete Protodescriptor classes
# =============================================================================
# region Concrete Protodescriptor classes


@declarator
class DataProtodescriptor(AssignableFieldProtodescriptor):
    """The protodescriptor class for data fields."""
    __handle__ = "data"
    index: bool = field(default=False, z=330)
    typekey: bool = field(default=False, z=301)
    key: bool = field(default=False, z=335)
    sequence: bool = field(default=False, z=340)
    timestamp: bool = field(default=False, z=345)
    start_time: bool = field(default=False, z=350)
    end_time: bool = field(default=False, z=355)
    period: bool = field(default=False, z=360)
    location: bool = field(default=False, z=365)
    geom: bool = field(default=False, z=370)
    gt: Real | Missing = field(default=MISSING, z=600)
    ge: Real | Missing = field(default=MISSING, z=610)
    lt: Real | Missing = field(default=MISSING, z=620)
    le: Real | Missing = field(default=MISSING, z=630)
    min_length: int | Missing = field(default=MISSING, z=640)
    max_length: int | Missing = field(default=MISSING, z=650)
    pattern: str | Missing = field(default=MISSING, z=660)

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
            DeclaratorReassignmentViolation.report(
                f"Data field '{self.name}' cannot be reassigned."
            )
            return previous
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            DeclaratorTypeConstraintViolation.report(
                f"Inline validation failed for data '{self.name}' with value: {value}."
            )
            return previous
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten data constraints only."""
        if type(override) is not type(self):
            DeclaratorOverrideViolation.report(
                "Data protodescriptors can only be overridden by the same class."
            )
            return
        if override == self:
            return
        self._validate_override_common(override)
        for attr in ("index", "required", "typekey", "key", "sequence", "timestamp",
                     "start_time", "end_time", "period", "location", "geom", "unique"):
            if not _tighten_bool(getattr(self, attr), getattr(override, attr)):
                DeclaratorOverrideViolation.report(
                    f"Data protodescriptor '{attr}' cannot be loosened."
                )
                return
        if not _tighten_lower(self.gt, self.ge, override.gt, override.ge):
            DeclaratorOverrideViolation.report(
                "Data protodescriptor lower bound cannot be loosened."
            )
            return
        if not _tighten_upper(self.lt, self.le, override.lt, override.le):
            DeclaratorOverrideViolation.report(
                "Data protodescriptor upper bound cannot be loosened."
            )
            return
        if is_not_missing(self.min_length) and is_not_missing(override.min_length):
            if override.min_length < self.min_length:
                DeclaratorOverrideViolation.report(
                    "Data protodescriptor min_length cannot be loosened."
                )
                return
        if is_not_missing(self.max_length) and is_not_missing(override.max_length):
            if override.max_length > self.max_length:
                DeclaratorOverrideViolation.report(
                    "Data protodescriptor max_length cannot be loosened."
                )
                return
        if is_not_missing(self.pattern) and is_not_missing(override.pattern):
            if override.pattern != self.pattern:
                DeclaratorOverrideViolation.report(
                    "Data protodescriptor pattern cannot be overridden."
                )
                return

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate a data binding in the context of a host type."""
        if not self.classvar:
            DeclaratorOverrideViolation.report(
                "Only data descriptors typed as class variables can be bound."
            )
            return False
        elif value is None:
            if not self.nullable:
                DeclaratorNullabilityViolation.report(
                    "Non-nullable data field cannot be bound to None."
                )
        elif self.type is not None:
            if not isinstance(value, self.type):
                if not (isinstance(value, type) and issubclass(value, self.type)):
                    DeclaratorTypeConstraintViolation.report(
                        "Cannot bind data field of incompatible type."
                    )
        return True

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "data field"


@declarator
class LinkProtodescriptor(AssignableFieldProtodescriptor, RelationProtodescriptor):
    """The protodescriptor class for link fields."""
    __handle__ = "link"
    key: bool = field(default=False, z=335)
    on_delete: Literal["restrict", "set_null", "cascade"] = field(default="restrict", z=375)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("on_delete", self.on_delete, str)
        self._validate_value_type("key", self.key, bool)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        DeclaratorOverrideViolation.report("Link fields are not prototype-bindable.")
        return previous

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten link constraints only."""
        if type(override) is not type(self):
            DeclaratorOverrideViolation.report(
                "Link protodescriptors can only be overridden by the same class."
            )
            return
        if override == self:
            return
        self._validate_override_common(override)
        for attr in ("required", "key", "unique"):
            if not _tighten_bool(getattr(self, attr), getattr(override, attr)):
                DeclaratorOverrideViolation.report(
                    f"Link protodescriptor '{attr}' cannot be loosened."
                )
                return
        order = {"cascade": 0, "set_null": 1, "restrict": 2}
        if order.get(override.on_delete, 0) < order.get(self.on_delete, 0):
            DeclaratorOverrideViolation.report(
                "Link protodescriptor on_delete cannot be loosened."
            )
            return

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "link field"


@declarator
class BackLinkProtodescriptor(IdentifiableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for backlink fields."""
    __handle__ = "backlink"
    via: type | Missing = field(default=MISSING, z=380)
    limit: int | Missing = field(default=MISSING, z=385)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.via):
            self._validate_value_type("via", self.via, type)
        if is_not_missing(self.limit):
            self._validate_value_type("limit", self.limit, int)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        DeclaratorOverrideViolation.report("Backlink fields are not prototype-bindable.")
        return previous

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten backlink constraints only."""
        if type(override) is not type(self):
            DeclaratorOverrideViolation.report(
                "Backlink protodescriptors can only be overridden by the same class."
            )
            return
        if override == self:
            return
        self._validate_override_common(override)
        if not _tighten_bool(self.unique, override.unique):
            DeclaratorOverrideViolation.report(
                "Backlink protodescriptor 'unique' cannot be loosened."
            )
            return
        if is_not_missing(self.via) and is_not_missing(override.via):
            if override.via != self.via:
                DeclaratorOverrideViolation.report(
                    "Backlink protodescriptor 'via' cannot be overridden."
                )
                return
        if is_not_missing(self.limit) and is_not_missing(override.limit):
            if override.limit > self.limit:
                DeclaratorOverrideViolation.report(
                    "Backlink protodescriptor limit cannot be loosened."
                )
                return

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "backlink field"


@declarator
class SelectionProtodescriptor(RelationProtodescriptor, CallableDeclarator[Callable]):
    """Protodescriptor for selection relations."""
    __handle__ = "selection"
    kind: Literal["sql", "ibis"] | Missing = field(default=MISSING, z=265)
    # Syntactic sugar for callable-based forms
    sql: Callable | Missing = field(default=MISSING, z=271)
    ibis: Callable | Missing = field(default=MISSING, z=272)
    # Field-expression-based form
    fx: Callable | str | Missing = field(default=MISSING, z=273)
    # Frame-based form
    source: Any | Missing = field(default=MISSING, z=266)
    key: Any | list[Any] | Missing = field(default=MISSING, z=281)
    time: Any | Missing = field(default=MISSING, z=282)
    space: Any | Missing = field(default=MISSING, z=283)
    filter: Callable | Any | Missing = field(default=MISSING, z=284)
    sort: str | list[str] | Missing = field(default=MISSING, z=285)
    limit: int | Missing = field(default=MISSING, z=286)
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
            DeclaratorTypeConstraintViolation.report(
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
                DeclaratorTypeConstraintViolation.report(
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
            DeclaratorTypeConstraintViolation.report(
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
                        DeclaratorTypeConstraintViolation.report(
                            "SelectionProtodescriptor.sort must be strings."
                        )
                else:
                    self._validate_value_type("sort", self.sort, str)
            if is_not_missing(self.limit):
                self._validate_value_type("limit", self.limit, int)

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "selection field"


@declarator
class DocumentProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for document fields."""
    __handle__ = "document"
    path: str | Missing = field(default=MISSING, z=295)
    format: str | Missing = field(default=MISSING, z=380)
    compression: str | Missing = field(default=MISSING, z=381)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.path):
            self._validate_value_type("path", self.path, str)
        if is_not_missing(self.format):
            self._validate_value_type("format", self.format, str)
        if is_not_missing(self.compression):
            self._validate_value_type("compression", self.compression, str)

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "document field"


@declarator
class FolderProtodescriptor(AssignableDeclarator, RelationProtodescriptor):
    """The protodescriptor class for folder fields."""
    __handle__ = "folder"
    path: str | Missing = field(default=MISSING, z=296)


    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.path):
            self._validate_value_type("path", self.path, str)

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "folder field"


@declarator
class StateProtodescriptor(RelationProtodescriptor):
    """Protodescriptor for dynamic state values."""
    __handle__ = "state"
    # Underlying time series
    source: type | Protodescriptor | Missing = field(default=MISSING, z=266)
    # Reduction policy
    reducer: str | Callable | Missing = field(default=MISSING, z=287)
    # Constraints
    max_lag: str | timedelta | Missing = field(default=MISSING, z=288)
    min_observations: int | Missing = field(default=MISSING, z=289)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.reducer):
            self._validate_value_type("reducer", self.reducer, (str, Callable))  # type: ignore[arg-type]
        if is_not_missing(self.max_lag):
            self._validate_value_type("max_lag", self.max_lag, (str, timedelta))  # type: ignore[arg-type]
        if is_not_missing(self.min_observations):
            self._validate_value_type("min_observations", self.min_observations, int)
            if self.min_observations < 1:
                DeclaratorTypeConstraintViolation.report(
                    "state.min_observations must be >= 1."
                )
                return

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "state field"


@declarator
class FieldExpressionProtodescriptor(FieldProtodescriptor, CallableDeclarator[Callable[[type], Any]]):  # noqa
    """The protodescriptor class for field expressions."""
    __handle__ = "fx"
    ref: str | Missing = field(default=MISSING, z=274)

    def __validate__(self) -> None:
        super().__validate__()
        if is_missing(self.callable) and is_missing(self.ref):
            DeclaratorTypeConstraintViolation.report(
                "FieldExpressionProtodescriptor requires either a 'callable' or 'ref' attribute."
            )
            return
        if is_not_missing(self.ref):
            self._validate_value_type("ref", self.ref, str)
        if is_not_missing(self.callable):
            self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "field expression"


@declarator
class FieldGroupProtodescriptor(FieldProtodescriptor, FieldEnumeration):
    """The protodescriptor class for field groups."""
    __handle__ = "fieldgroup"

    def __validate__(self) -> None:
        super().__validate__()
        if not self.members:
            DeclaratorTypeConstraintViolation.report(
                "FieldGroupProtodescriptor must have at least one member."
            )
            return

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "field group"


@declarator
class FieldBlockProtodescriptor(FieldProtodescriptor):
    """The protodescriptor class for field blocks."""
    __handle__ = "fieldblock"

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "field block"


@declarator
class MetricProtodescriptor(FieldProtodescriptor, CallableDeclarator[Callable[[type], Any]]):  # noqa
    """The protodescriptor class for metric fields."""
    __handle__ = "metric"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]

    @classproperty
    def typename(cls) -> str:
        """A message-friendly shorthand for the protodescriptor's type."""
        return "metric field"


@declarator
class ParserDeclarator(ConstructorDeclarator[Callable[[type, Any], Any]], FieldEnumeration):  # noqa
    """The declarator class for field parsers."""
    __handle__ = "parser"
    element_wise: bool = field(default=False, z=291)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("element_wise", self.element_wise, bool)


@declarator
class ValidatorDeclarator(ConstructorDeclarator[Callable[[type, Any], bool]], FieldEnumeration):  # noqa
    """The declarator class for field validators."""
    __handle__ = "validator"
    element_wise: bool = field(default=False, z=291)

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("element_wise", self.element_wise, bool)


# Schema Declarators

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
    kind: str | Missing = field(default=MISSING, z=292)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.kind):
            self._validate_value_type("kind", self.kind, str)


@declarator
class PartitionDeclarator(SchemaDeclarator):
    """The declarator class for schema partitions."""
    __handle__ = "partition"
    key: FieldProtodescriptor = field(z=293)
    scheme: Literal["range", "categorical", "hash", "time"] = field(z=294)
    # MISSING → not specified (inherit / infer)
    # None → scheme-defined implicit bucketing (categorical, hash)
    buckets: None | int | tuple | list | str | Missing = field(default=MISSING, z=296)

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
    template: str | Missing = field(default=MISSING, z=295)

    def __validate__(self) -> None:
        super().__validate__()
        if is_not_missing(self.template):
            self._validate_value_type("template", self.template, str)


@declarator
class SortDeclarator(FieldEnumeration, SchemaDeclarator):
    """The declarator class for schema sort orders."""
    __handle__ = "sort"
    sort_directions: Literal["asc", "desc"] | list[Literal["asc", "desc"]] | Missing = field(default=MISSING, z=297)  # noqa

    def __validate__(self) -> None:
        super().__validate__()
        if not self.members:
            DeclaratorTypeConstraintViolation.report(
                "SortDeclarator must have at least one member."
            )
            return
        if is_not_missing(self.sort_directions):
            if not isinstance(self.sort_directions, (str, list)):
                DeclaratorTypeConstraintViolation.report(
                    "SortDeclarator.sort_directions must be 'asc' or 'desc' or a list thereof."
                )
                return
            if isinstance(self.sort_directions, str):
                if self.sort_directions not in {"asc", "desc"}:
                    DeclaratorTypeConstraintViolation.report(
                        "SortDeclarator.sort_directions must be 'asc' or 'desc'."
                    )
                    return
            else:
                invalid = [v for v in self.sort_directions if v not in {"asc", "desc"}]
                if invalid:
                    DeclaratorTypeConstraintViolation.report(
                        "SortDeclarator.sort_directions must be 'asc' or 'desc'."
                    )
                    return
                if not len(self.sort_directions) == len(self.members):
                    DeclaratorTypeConstraintViolation.report(
                        "SortDeclarator.sort_directions length must match members length."
                    )
                    return

# endregion
