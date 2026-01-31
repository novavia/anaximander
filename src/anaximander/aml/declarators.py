"""Defines the Declarator class and its subclasses."""

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
    dataclass_transform,
)

from attrs import define, field

from ..utils.meta import classproperty

# endregion

# =============================================================================
# Helpers and globals
# =============================================================================
# region Constants


# Helpers for Declarator classes
DT = TypeVar("DT", bound=type)  # Declarator-type type variable
T = TypeVar("T")

@dataclass_transform(kw_only_default=True, field_specifiers=(field,))
def declarator(cls: DT) -> DT:
    """Apply attrs define for declarator classes with a preserved init signature."""
    return define(cls, slots=False, frozen=True, kw_only=True)


# Missing sentinel
class _MissingSentinel:
    """Sentinel object representing the absence of a declarative value.

    This is distinct from None, False, 0, or empty containers.
    It is used to indicate that an attribute was not declared.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        raise TypeError("MISSING has no truth value")

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other

    def __hash__(self) -> int:
        return id(self)

    def __reduce__(self):
        # Ensures singleton behavior across pickling
        return "MISSING"

Missing: TypeAlias = _MissingSentinel

# The one and only instance
MISSING = _MissingSentinel()

def is_missing(value: object) -> TypeGuard[Missing]:
    """Whether a value is the MISSING sentinel."""
    return value is MISSING

def is_not_missing(value: T | Missing) -> TypeGuard[T]:
    """Whether a value is not the MISSING sentinel."""
    return value is not MISSING


# Helper functions for declarator hooks
def _tighten_type(base: type | None, override: type | None) -> bool:
    """Whether a type override is a monotone tightening."""
    if base is None or override is None:
        return True
    try:
        return issubclass(override, base)
    except TypeError:
        return False


def _tighten_nullable(base: bool | None, override: bool | None) -> bool:
    """Whether a nullable override is monotone tightening."""
    if base is None or override is None:
        return True
    return not (base is False and override is True)


def _tighten_classvar(base: bool | None, override: bool | None) -> bool:
    """Whether a classvar override preserves classvar semantics."""
    if base is None or override is None:
        return True
    return base == override


def _tighten_bool(base: bool, override: bool) -> bool:
    """Whether a boolean override is monotone tightening."""
    return not base or override


def _tighten_bound_value(value: Any, previous: Any) -> bool:
    """Whether a bound value is a monotone tightening of its predecessor."""
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

# endregion

# =============================================================================
# Declarator base class
# =============================================================================
# region Declarator base class


@define(frozen=True)
class DeclarativeTypeKey:
    """A unique key for identifying declarative types."""
    project: str
    module: str
    name: str


class DeclarativeProtocol(Protocol):
    """A protocol for declarative types with unique keys."""
    __key__: ClassVar[DeclarativeTypeKey]

Declarative = type[DeclarativeProtocol]


@define(frozen=True)
class DeclaratorKey:
    """A unique key for identifying declarators."""
    owner: DeclarativeTypeKey
    ordinal: int


class DeclaratorConfig(Protocol):
    """A protocol for extending declarator configuration."""

    def resolve(self, **context) -> Mapping[str, Any]: ...


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
    name: str = field(init=False, default=None)  # Attribute or key name this declarator is assigned to # noqa
    owner: Declarative = field(init=False, default=None)  # Owning class of this declarator
    ordinal: int = field(init=False, default=None)  # Index of this declarator within the owning class # noqa
    __ast__: ast.AST | None = field(init=False, default=None)  # AST node that declared this declarator # noqa

    # Init-time fields (immutable)
    doc: str | Missing = field(default=MISSING)  # Optional documentation string # noqa
    config: Mapping[str, ConfigValue] | Missing = field(factory=dict)  # Extraneous declarator configuration # noqa

    @property
    def __key__(self) -> DeclaratorKey:
        """A unique key for identifying this declarator."""
        if self.owner is None or self.ordinal is None:
            raise RuntimeError("Declarator must be bound to a class before accessing its key.")
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
    def dtype(cls):
        """A message-friendly shorthand for the declarator's type."""
        return cls.__handle__ or cls.__name__

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values in the domain namespace."""
        return False

    def __attrs_post_init__(self) -> None:
        """Post-initialization processing for the declarator."""
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
                raise TypeError(
                    "All elements of __reserved_patterns__ must be instances of str or re.Pattern"
                )
            cls.__reserved_patterns__ = reserved_patterns | set(cls.__reserved_patterns__)
        else:
            cls.__reserved_patterns__ = reserved_patterns
        # Register and resolve handles
        if cls.__handle__ == "" and any(b.__handle__ != "" for b in base_declarators):  # noqa
            raise ValueError("Declarator subclasses cannot define an empty __handle__ if any base class does not.")  # noqa
        if cls.__handle__ != "":
            if cls.__handle__ in cls.__handles__:
                if cls.__handles__[cls.__handle__] not in cls.mro():
                    raise ValueError(f"Declarator handle '{cls.__handle__}' is already registered.")  # noqa
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
            raise RuntimeError(
                f"{self.__class__.__name__}.{attr} is already set."
            )
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
            raise TypeError(
                f"{self.__class__.__name__}.{attr} is missing."
            )

        if not isinstance(value, expected):
            if isinstance(expected, tuple):
                expected_name = " | ".join(t.__name__ for t in expected)
            else:
                expected_name = expected.__name__
            raise TypeError(
                f"{self.__class__.__name__}.{attr} must be {expected_name}, "
                f"got {type(value).__name__}."
            )

    def __set_name__(self, owner: type, name: str):
        """Attach the name and owner class to this declarator."""
        self._validate_value_type("name", name, str)
        self._validate_value_type("owner", owner, type)
        forbidden_names = {
            pattern for pattern in self.__reserved_patterns__ if isinstance(pattern, str)
        }
        forbidden_patterns = {
            pattern for pattern in self.__reserved_patterns__ if isinstance(pattern, re.Pattern)
        }
        if name in forbidden_names:
            msg = f"Cannot use reserved name {name} for declarator or type {self.dtype}."
            raise ValueError(msg)
        if any(pattern.fullmatch(name) for pattern in forbidden_patterns):
            msg = f"Cannot use reserved name {name} for declarator or type {self.dtype}."
            raise ValueError(msg)
        self._set_once("name", name, treat_none_as_unset=True)
        self._set_once("owner", owner, treat_none_as_unset=True)

    def __set_ast__(self, node: ast.AST | None) -> None:
        """Attach the AST node that declared this declarator (if any)."""
        self._validate_value_type("__ast__", node, (ast.AST , type(None)))
        self._set_once("__ast__", node, treat_none_as_unset=True)

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
        raise AttributeError(
            f"Declarator of type {self.__class__.__name__} cannot be bound to a value."
        )

    @abstractmethod
    def __override__(self, override: "Declarator") -> None:
        """Hook called when this declarator is overridden in a subclass.

        This method can be overridden by subclasses to customize the override behavior.
        The default implementation raises an AttributeError.
        """
        if override == self:
            return
        raise AttributeError(f"Declarator of type {self.__class__.__name__} cannot be overridden.")

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Hook called to validate a bound value in the context of a hosting class.

        Unlike __bind__, which is called when the binding occurs and does not use context,
        this method is designed to be called at AML module finalization.
        This method can be overridden by subclasses to implement custom validation logic.
        The default implementation returns True.
        """
        return True


@declarator
class AnnotatableDeclarator(Declarator):
    """Base class for declarators that can be annotated with type information."""
    annotation: str | None = field(init=False, default=None)  # Literal type annotation as a string  # noqa
    hint: Any | None = field(init=False, default=None)  # Evaluated type hint object  # noqa
    type: builtins.type | None = field(init=False, default=None)  # Evaluated type annotation  # noqa
    nullable: bool | None = field(init=False, default=None)  # Whether the type is nullable  # noqa
    classvar: bool | None = field(init=False, default=None)  # Whether the type is a ClassVar  # noqa
    __types__: ClassVar[tuple[builtins.type, ...]] = ()  # Admissible types for this annotatable declarator  # noqa

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Check that __types__ are tightening the admissible types from base classes.
        base_annotatables = (b for b in cls.__bases__ if issubclass(b, AnnotatableDeclarator))
        base_types = tuple(chain.from_iterable(b.__types__ for b in base_annotatables))
        if "__types__" in vars(cls):
            types: tuple[type, ...] = cls.__types__
            try:
                assert all(issubclass(t, base_type) for t in types for base_type in base_types)
            except AssertionError:
                raise TypeError(
                    "__types__ must only contain types that are subclasses of all "
                    + "admissible types from base classes."
                )

    def __validate_type__(self, type_: type) -> bool:
        if not self.__types__:
            return True
        try:
            return issubclass(type_, self.__types__)
        except TypeError:
            return False

    def __validate_annotation__(self, annotation: str | None, hint: Any | None) -> None:
        """Validate a resolved annotation for this declarator."""

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
        self.__validate_annotation__(annotation, hint)
        if type_ is not None and not self.__validate_type__(type_):
            declarator = self.name
            owner_name = self.owner.__name__
            msg = (
                f"Incompatible annotation {annotation} supplied to {declarator} declarator "
                + f"of {owner_name}."
            )
            raise TypeError(msg)
        self._set_once("annotation", annotation, treat_none_as_unset=True)
        self._set_once("hint", hint, treat_none_as_unset=True)
        self._set_once("type", type_, treat_none_as_unset=True)
        self._set_once("nullable", nullable, treat_none_as_unset=True)
        self._set_once("classvar", classvar, treat_none_as_unset=True)


@declarator
class IdentifiableDeclarator(AnnotatableDeclarator):
    """Base class for declarators of attributes that can uniquely identify an instance."""
    unique: bool = field(default=False)  # Whether this declarator uniquely identifies an instance  # noqa

    def __validate__(self) -> None:
        self._validate_value_type("unique", self.unique, bool)
        return super().__validate__()


@declarator
class AssignableDeclarator(AnnotatableDeclarator):
    """Base class for declarators of attributes that receive their value through assignment."""
    default: Any = field(default=MISSING)
    factory: Callable[[], Any] | Missing = field(default=MISSING)
    # This is a fail-quick optional inline validator that takes a value as its only argument
    # It is intended to be called in the __bind__ hook to validate assigned values
    validator: Callable[[Any], bool] | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        if is_not_missing(self.factory):
            self._validate_value_type("factory", self.factory, Callable)  # type: ignore[arg-type]
        if is_not_missing(self.validator):
            self._validate_value_type("validator", self.validator, Callable)  # type: ignore[arg-type]
        return super().__validate__()


@declarator
class CallableDeclarator[C: Callable](Declarator):
    """A mixin class for declarators that wrap callables."""
    callable: C | Missing = field(default=MISSING)

    def __validate__(self) -> None:
        if is_not_missing(self.callable):
            self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]
        return super().__validate__()


@declarator
class EnumerationDeclarator(Declarator):
    """A mixin class for declarators that reference a list of declarators by name."""
    members: tuple[str, ...] = field(factory=tuple)
    __member_types__: ClassVar[tuple[type[Declarator], ...]] = ()  # Admissible member types

    def __validate__(self) -> None:
        self._validate_value_type("members", self.members, tuple)
        if any(not isinstance(member, str) for member in self.members):
            raise TypeError("EnumerationDeclarator.members must be a tuple of strings.")
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
                raise TypeError(
                    "__member_types__ must only contain types that are subclasses of all "
                    + "admissible types from base classes."
                )


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

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate a metadata binding in the context of a host type."""
        if is_missing(value):
            if self.default is not MISSING:
                return True
            else:
                raise ValueError("Metadata binding cannot be missing.")
        elif value is None:
            if not self.nullable:
                raise ValueError("Non-nullable metadata cannot be bound to None.")
        elif self.type is not None:
            if not isinstance(value, self.type):
                if not (isinstance(value, type) and issubclass(value, self.type)):
                    raise TypeError("Cannot bind metadata of incompatible type.")
        return True


@declarator
class MetadataDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for metadata fields."""
    __handle__ = "metadata"
    domain: bool = field(default=False)  # Whether this metadata is part of the domain schema  # noqa

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values."""
        return self.domain

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("domain", self.domain, bool)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind a metadata value, enforcing monotone tightening and inline validation."""
        if is_not_missing(previous) and value != previous:
            if not _tighten_bound_value(value, previous):
                raise AttributeError(
                    f"Metadata '{self.name}' cannot be reassigned with a looser value."
                )
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            raise ValueError(
                f"Inline validation failed for metadata '{self.name}' with value: {value}"
            )
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten type constraints only."""
        if type(override) is not type(self):
            raise AttributeError("Metadata declarators can only be overridden by the same class.")
        if override == self:
            return
        if self.domain != override.domain:
            raise AttributeError("Metadata declarator 'domain' cannot be overridden.")
        if not _tighten_type(self.type, override.type):
            raise AttributeError("Metadata declarator type cannot be loosened.")
        if not _tighten_nullable(self.nullable, override.nullable):
            raise AttributeError("Metadata declarator nullability cannot be loosened.")
        if not _tighten_classvar(self.classvar, override.classvar):
            raise AttributeError("Metadata declarator classvar cannot be overridden.")


@declarator
class OptionDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for option fields."""
    __handle__ = "option"


@declarator
class NxFieldDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"
    fieldtype: type = field()

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("fieldtype", self.fieldtype, type)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind an nxfield value, disallowing reassignment in derived prototypes."""
        if is_not_missing(previous) and value != previous:
            raise AttributeError(f"NxField '{self.name}' cannot be reassigned.")
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            raise ValueError(
                f"Inline validation failed for nxfield '{self.name}' with value: {value}"
            )
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten fieldtype/type constraints only."""
        if type(override) is not type(self):
            raise AttributeError("NxField declarators can only be overridden by the same class.")
        if override == self:
            return
        if not _tighten_type(self.fieldtype, override.fieldtype):
            raise AttributeError("NxField declarator fieldtype cannot be loosened.")
        if not _tighten_type(self.type, override.type):
            raise AttributeError("NxField declarator type cannot be loosened.")
        if not _tighten_nullable(self.nullable, override.nullable):
            raise AttributeError("NxField declarator nullability cannot be loosened.")
        if not _tighten_classvar(self.classvar, override.classvar):
            raise AttributeError("NxField declarator classvar cannot be overridden.")

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate an nxfield binding in the context of a host type."""
        # TODO: verify that value is a string that matches a field of host of type fieldtype
        return True


# Metavalidators

@declarator
class MetaValidator(Metadescriptor):
    """A common base class for metadescriptor validators."""
    __handle__ = "metavalidator"


@declarator
class PrototypeValidator(CallableDeclarator[Callable[[type], bool]], MetaValidator):
    """The metadescriptor class for prototype validators."""
    __handle__ = "prototype_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class MetadataValidator(EnumerationCallableDeclarator[Callable[[type, Any], bool]], MetaValidator):  # noqa
    """The metadescriptor class for metadata validators."""
    __handle__ = "metadata_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class OptionValidator(EnumerationCallableDeclarator[Callable[[type, Any], bool]], MetaValidator):  # noqa
    """The metadescriptor class for option validators."""
    __handle__ = "option_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class NxFieldValidator(EnumerationCallableDeclarator[Callable[[type, Any], bool]], MetaValidator):  # noqa
    """The metadescriptor class for nxfield validators."""
    __handle__ = "nxfield_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


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
    source: type | Protodescriptor | Missing = field(default=MISSING)
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
class FieldExpressionProtodescriptor(FieldProtodescriptor, CallableDeclarator[Callable[[type], Any]]):  # noqa
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
