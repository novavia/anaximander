"""Defines the Declarator class and declarative metaclass.

These artifacts form the foundation for declarative type definitions in AML.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import ast
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from functools import partial
from itertools import chain, count
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    Protocol,
    Self,
    TypeVar,
    dataclass_transform,
    get_args,
)

import yaml
from attrs import define, field

from ..utils.meta import classproperty

# endregion

# =============================================================================
# Constants
# =============================================================================
# region Constants


DT = TypeVar("DT", bound=type)  # Declarator-type type variable


@dataclass_transform(kw_only_default=True, field_specifiers=(field,))
def declarator(cls: DT) -> DT:
    """Apply attrs define for declarator classes with a preserved init signature."""
    return define(cls, slots=False, frozen=True, kw_only=True)

DECLARATIVE_NAMESPACE: ContextVar[Optional["DeclarativeNamespace"]] = ContextVar("DECLARATIVE_NAMESPACE", default=None)  # noqa

type Assignment = ast.Assign | ast.AnnAssign

# Sentinel value for unspecified defaults
class _MissingSentinel:
    """Unique sentinel for unspecified defaults."""
    def __repr__(self) -> str:
        return "MISSING"

MISSING: _MissingSentinel = _MissingSentinel()

# endregion

# =============================================================================
# Declarator base class
# =============================================================================
# region Declarator base class


@define(frozen=True)
class DeclaratorKey:
    """A unique key for identifying declarators."""
    owner: "DeclarativeTypeKey"
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
    owner: "declarative" = field(init=False, default=None)  # Owning class of this declarator
    ordinal: int = field(init=False, default=None)  # Index of this declarator within the owning class # noqa
    __ast__: ast.AST | None | _MissingSentinel = field(init=False, default=MISSING)  # AST node that declared this declarator  # noqa

    # Init-time fields (immutable)
    doc: str | None = field(default=None)  # Optional documentation string
    config: Mapping[str, ConfigValue] = field(factory=dict)  # Extraneous declarator configuration

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

    def __attrs_post_init__(self) -> None:
        """Post-initialization processing for the declarator."""
        dns = DECLARATIVE_NAMESPACE.get()
        if dns is not None:
            dns.register_declaration(self)
        # Freeze config to prevent accidental mutation.
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

    def _set_once(self, attr: str, value: Any, *, treat_none_as_unset: bool = True) -> None:
        """Internal backdoor: set a frozen attribute once (or idempotently)."""
        current = getattr(self, attr)
        if current is MISSING or (treat_none_as_unset and current is None):
            object.__setattr__(self, attr, value)
            return
        if current != value:
            raise RuntimeError(f"{self.__class__.__name__}.{attr} is already set.")
        object.__setattr__(self, attr, value)

    def __set_name__(self, owner: type, name: str):
        """Attach the name and owner class to this declarator."""
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
        self._set_once("name", name)
        self._set_once("owner", owner)

    def __set_ast__(self, node: ast.AST | None) -> None:
        """Attach the AST node that declared this declarator (if any)."""
        self._set_once("__ast__", node, treat_none_as_unset=False)

    def __validate__(self) -> None:
        """Validate this declarator after it has been bound to a namespace.

        This method is called by the declarative metaclass after the declarator
        has been assigned to a class attribute, but before the class is finalized.
        Its role is to validate the declarator's attributes and configuration,
        absent any context. Subclasses can override this method to implement custom
        validation logic.
        """
        pass

    @abstractmethod
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
        this method is designed to be called when the host class is finalized.
        This method can be overridden by subclasses to implement custom validation logic.
        The default implementation returns True.
        """
        return True


@declarator
class AnnotatableDeclarator(Declarator):
    """Base class for declarators that can be annotated with type information."""
    annotation: str | None = field(init=False, default=None)  # Literal type annotation as a string  # noqa
    type: type | None = field(init=False, default=None)  # Evaluated type annotation  # noqa
    nullable: bool | None = field(init=False, default=None)  # Whether the type is nullable  # noqa
    classvar: bool | None = field(init=False, default=None)  # Whether the type is a ClassVar  # noqa
    __types__: ClassVar[tuple[type, ...]] = ()  # Admissible types for this annotatable declarator

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

    @abstractmethod
    def __validate_type__(self, type: Any) -> bool:
        if not self.__types__:
            return True
        return issubclass(type, self.__types__)

    def __set_type__(
        self,
        annotation: str,
        type: Any | None,
        nullable: bool | None,
        classvar: bool | None = None,
    ):
        """Sets the type by supplying annotation, evaluated type, nullability and classvar."""
        if type is not None and not self.__validate_type__(type):
            declarator = self.name
            owner_name = self.owner.__name__
            msg = (
                f"Incompatible annotation {annotation} supplied to {declarator} declarator "
                + f"of {owner_name}."
            )
            raise TypeError(msg)
        self._set_once("annotation", annotation)
        self._set_once("type", type)
        self._set_once("nullable", nullable)
        self._set_once("classvar", classvar)


@declarator
class IdentifiableDeclarator(AnnotatableDeclarator):
    """Base class for declarators of attributes that can uniquely identify an instance."""
    unique: bool | None = field(init=False, default=None)

    def __set_unique__(self, unique: bool):
        self._set_once("unique", unique)


@declarator
class AssignableDeclarator(IdentifiableDeclarator):
    """Base class for declarators of attributes that receive their value through assignment."""
    default: Any = field(default=MISSING)
    factory: Callable[[], Any] | _MissingSentinel = field(default=MISSING)
    # This is a fail-quick optional inline validator that takes a value as its only argument
    # It is intended to be called in the __bind__ hook to validate assigned values
    validator: Callable[[Any], bool] | None = field(default=None)


@declarator
class CallableDeclarator(Declarator):
    """A mixin class for declarators that wrap callables."""
    callable: Callable | None = field(default=None)


@declarator
class EnumerationDeclarator(Declarator):
    """A mixin class for declarators that reference a list of declarators by name."""
    members: tuple[str, ...] = field(factory=tuple)
    __member_types__: ClassVar[tuple[type[Declarator], ...]] = ()  # Admissible member types

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
class EnumerationCallableDeclarator(CallableDeclarator, EnumerationDeclarator):
    """A mixin class for callable declarators that reference a list of declarators by name."""
    pass

# endregion

# =============================================================================
# Declarative metaclass
# =============================================================================
# region Declarative metaclass


@define(frozen=True)
class DeclarativeTypeKey:
    """A unique key for identifying declarative types."""
    project: str
    module: str
    name: str


class DeclarativeNamespace(dict[str, Any]):
    """Collects class body declarations for a declarative type."""
    context_token: Token

    def __init__(self, *, strict: bool = True, bindable_names: list[str] | None = None):
        super().__init__(__declarations__=dict(), __bindings__=dict())
        self.strict = strict
        self.bindable_names = bindable_names or []
        self.declaration_index = count(start=1)

    def __set_item__(self, key: str, value: Any) -> None:
        if key in self:
            raise RuntimeError(f"Cannot redefine name '{key}' in declarative namespace.")
        if key in self.bindable_names:
            self.register_binding(key, value)
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        raise RuntimeError("Cannot delete items from a declarative namespace.")

    def register_declaration(self, declarator: Declarator, *, name: str | None = None) -> None:
        """Register a declarator in this namespace.

        name is optional and only used when the declarator is registered through a namespace
        with a .declare() method. Otherwise the declarator is not given a name yet as it is
        set by the __set_name__ hook after the class body is executed.
        """
        declarations: dict[int, Declarator] = self["__declarations__"]
        if not isinstance(declarator, Declarator):
            raise TypeError(f"Expected a Declarator instance, got {declarator}.")
        if name is not None:
            # sentinel to fail quick if name contains a forbidden dot
            if "." in name:
                raise ValueError("Declarator names cannot contain '.'.")
            # next we check for name conflicts, starting with a cursory lookup
            # and then refining it at the namespace level if warranted
            if name in (d.name for d in declarations.values()):
                declarator_handle = declarator.handle
                matches = [d for d in declarations.values() if d.name == name]
                if any(d.handle == declarator_handle for d in matches):
                    msg = f"Duplicate declaration for name '{name}'"
                    if declarator_handle:
                        msg += f" and handle '{declarator_handle}'."
                    raise KeyError(msg)
            declarator._set_once("name", name)
        if declarator in declarations.values():
            return
        ordinal = next(self.declaration_index)
        declarations[ordinal] = declarator
        declarator._set_once("ordinal", ordinal)

    def register_binding(self, key: str, value: Any, *, handle: str | None= None) -> None:
        """Register a binding in this namespace.

        The handle parameter specifies the namespace handle under which the binding is registered.
        It is optional and defaults to None, which indicates the declarative namespace itself.
        The handle is passed when the binding is registered through a declarator namespace
        using the .bind() method. If it belongs to one of the metadescriptor namespaces,
        specifically "metadata", "option", or "nxfield", the binding key is prefixed with
        the handle and a dot, reflecting the namespace context.
        """
        bindings: dict[str, Any] = self["__bindings__"]
        if handle in {"metadata", "option", "nxfield"}:
            binding_key = f"{handle}.{key}"
            if binding_key in bindings:
                raise KeyError(f"Binding '{key}' is already registered in namespace '{handle}'.")
        else:
            binding_key = key
            if binding_key in bindings:
                raise KeyError(f"Binding '{key}' is already registered in this namespace.")
        bindings[binding_key] = value

    def close(self) -> None:
        """Close the namespace, preventing further modifications."""
        try:
            if self.strict:
                # All non-dunder attributes in the namespace must be either declarations or bindings.  #noqa
                for key in self:
                    if not key.startswith("__") and key not in self["__declarations__"] and key not in self["__bindings__"]:  # noqa
                        raise RuntimeError(
                            f"Name '{key}' is neither a declaration nor a binding in strict mode."
                        )
        finally:
            DECLARATIVE_NAMESPACE.reset(self.context_token)


class declarative(type):
    """Metaclass for declarative types."""

    __project__: ClassVar[str] = "anaximander"  # Project name for this declarative type
    __declarations__: dict[int, Declarator]  # Declarators declared in this type
    __bindings__: dict[str, Any]  # Bindings made in this type
    __strict__: bool = False  # Whether this type uses strict declaration rules
    __ast__: ast.ClassDef | None  # Holds the type's parsed abstract syntax tree

    @property
    def __key__(cls) -> DeclarativeTypeKey:
        """A unique key for identifying this declarative type."""
        module = cls.__module__
        name = cls.__qualname__
        return DeclarativeTypeKey(
            project=cls.__project__,
            module=module,
            name=name,
        )

    @property
    @abstractmethod
    def _bindable_names(cls) -> list[str]:
        """Names that can be bound in this declarative type's subclasses."""
        return []

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs) -> DeclarativeNamespace:
        """Collects declarations, assignments and containers in the class body."""
        declarative_parents = [b for b in bases if isinstance(b, declarative)]
        bindable_names = [*chain(*(b._bindable_names for b in declarative_parents))]
        namespace = DeclarativeNamespace(
            strict=mcls.__strict__,
            bindable_names=bindable_names,
        )
        token = DECLARATIVE_NAMESPACE.set(namespace)
        namespace.context_token = token
        return namespace

    def __new__(mcls, name, bases, namespace: DeclarativeNamespace, **kwargs):
        """Create the new declarative type, processing its declarations and bindings."""
        try:
            cls = super().__new__(mcls, name, bases, dict(namespace))
        finally:
            namespace.close()
        for declarator in cls.__declarations__.values():
            if declarator.owner is None:
                if declarator.name is None:
                    raise RuntimeError("Unnamed declarator registered outside class assignment.")
                declarator.__set_name__(cls, declarator.name)
        # Runs declarator validation hooks
        for declarator in cls.__declarations__.values():
            declarator.__validate__()
        return cls


# endregion

# =============================================================================
# Registry classes
# =============================================================================
# region Registry classes


class Registry[T](Mapping[str, T]):
    """A base registry class for declarator and binding registries."""

    def __init__(self,
        data: Mapping[str, T] | Iterable[tuple[str, T]] | None = None,
        **kwargs: T
    ) -> None:
        """Initialize the data with an optional mapping."""
        self._data: dict[str, T] = dict(data or {}, **kwargs)

    def __getitem__(self, key: str) -> T:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"Key {key} not found in registry.") from None

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __copy__(self) -> Self:
        """Create a shallow copy of the registry."""
        new_registry = self.__class__()
        new_registry._data = self._data.copy()
        return new_registry

    def copy(self) -> Self:
        """Create a shallow copy of the registry."""
        return self.__copy__()

    @abstractmethod
    def register(self, key: str, item: T) -> None:
        """Register an item with the given key."""
        raise NotImplementedError

    def update(self, other: Mapping[str, T]) -> None:
        """Update the registry with items from another mapping."""
        for key, item in other.items():
            self.register(key, item)

    def to_dict(self) -> dict[str, T]:
        """Convert to a plain dict suitable for serialization."""
        return dict(self._data)

    def to_yaml(self) -> str:
        """YAML-style pretty print."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def __str__(self) -> str:
        return self.to_yaml()


class DeclaratorRegistry[D: Declarator](Registry[D]):
    """A specialized registry for declarators."""

    __types__: ClassVar[tuple[type[Declarator], ...]]  # Admissible declarator types

    def __init_subclass__(cls):
        """Assumes the first base is the base declarator registry."""
        super().__init_subclass__()
        base: type[DeclaratorRegistry]= cls.__bases__[0]
        super_types = base.__types__
        if "__types__" in cls.__dict__:
            types: tuple[type[Declarator], ...] = cls.__types__
        else:
            types = ()
            # Case 1: Subclass is generic and narrows the bound (e.g., [D: FieldDescriptor])
            if type_parameters := getattr(cls, "__type_params__", ()):
                param = type_parameters[0]
                types = (getattr(param, "__bound__"),)
                cls.__types__ = types
            # Case 2: Subclass specializes the base (e.g., DeclaratorRegistry[FieldDescriptor])
            elif (original_bases := getattr(cls, "__orig_bases__", ())):
                # Assume first original base is the base with type arguments
                base = original_bases[0]
                type_arguments = get_args(base)
                if type_arguments:
                    types = (type_arguments[0],)
                    cls.__types__ = types
        if types:
            try:
                assert all(issubclass(t, Declarator) for t in types)
                assert all(any(issubclass(t, u) for u in super_types) for t in types)
            except AssertionError:
                raise TypeError("All types in __types__ must be subclasses of Declarator and " +
                                "also subclasses of the base registry's __types__.")

    def __init__(self,
        data: Mapping[str, D] | Iterable[tuple[str, D]] | None = None,
        **kwargs: D
    ) -> None:
        """Initialize the declarator registry with an optional mapping or iterable of pairs."""
        super().__init__(data, **kwargs)

    def __copy__(self) -> Self:
        """Create a shallow copy of the registry."""
        new_registry = self.__class__()
        new_registry._data = self._data.copy()
        return new_registry

    def register(self, key: str, item: D) -> None:
        """Registers a declarator."""
        if not isinstance(item, self.__types__):
            type_names = ", ".join(t.__name__ for t in self.__types__)
            raise TypeError(f"Declarator must be one of types: {type_names}")
        registered = self._data.get(key)
        if registered is None:
            self._data[key] = item
        elif registered == item:
            return
        else:
            try:
                self._data[key].__override__(item)
            except AttributeError as e:
                raise RuntimeError(f"Cannot override declarator '{key}': {e}") from e

    def filter(self, *args: type | str) -> Self:
        """Yield declarators that match the given types or handles."""
        if not args:
            return self.__copy__()
        data = self._data
        types = []
        for arg in args:
            if isinstance(arg, type):
                types.append(arg)
            elif isinstance(arg, str):
                try:
                    types.append(Declarator.__handles__[arg])
                except KeyError:
                    raise ValueError(f"Unknown declarator handle: {arg}") from None
            else:
                raise TypeError("Arguments must be types or declarator handles (strings).")
        if not all(any(issubclass(t, __t__) for __t__ in self.__types__) for t in types):
            type_names = ", ".join(t.__name__ for t in self.__types__)
            raise TypeError(f"Arguments must be one of types: {type_names} or their handles.")
        return self.__class__({k: v for k, v in data.items() if isinstance(v, tuple(types))})


class BindingRegistry[D: Declarator](Registry[Any]):
    """Registry for bindings."""

    def __init__(self,
        data: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        *,
        _declarators: DeclaratorRegistry[D],
        **kwargs: Any
    ) -> None:
        """Initialize the binding registry with an optional declarator registry."""
        if "_declarators" in kwargs:
            raise ValueError("Declarators cannot be passed as a keyword argument.")
        self._declarators = _declarators
        bindings = dict(data or {}, **kwargs)
        self._data: dict[str, Any] = {}
        for key, item in bindings.items():
            self.register(key, item)

    @property
    def declarators(self) -> DeclaratorRegistry[D] :
        """Return the referenced declarator registry, or None if it has been garbage collected."""
        return self._declarators

    def __copy__(self) -> Self:
        """Create a shallow copy of the binding registry."""
        new_registry = self.__class__(_declarators=self.declarators)
        new_registry._data = self._data.copy()
        return new_registry

    def register(self, key: str, item: Any) -> None:
        """Register a binding."""
        declarators = self.declarators
        if declarators is None:
            raise RuntimeError("Cannot register binding: declarator registry has been garbage collected.")  # noqa
        try:
            declarator: D = declarators[key]
        except KeyError:
            raise KeyError(f"No declarator found for binding '{key}'.")
        binding = self._data.get(key, MISSING)
        if item == binding:
            return
        else:
            try:
                declarator.__bind__(item, binding)
            except AttributeError as e:
                raise RuntimeError(f"Cannot bind value to declarator '{key}': {e}") from e
            else:
                self._data[key] = item


class MultiRegistry(Mapping[str, Registry[Any]]):
    """A mapping that organizes multiple registries by namespace or rubric."""

    __namespaces__: ClassVar[set[str]] = set()  # Set of valid registry namespaces

    def __init__(self, data: Mapping[str, Registry[Any]] | Iterable[tuple[str, Registry[Any]]] | None = None, **kwargs: Registry[Any]):  # noqa
        """Initialize the multi-registry with an optional mapping or iterable of pairs."""
        self._data: dict[str, Registry[Any]] = dict(data or {}, **kwargs)

    def __getitem__(self, key: str) -> Registry[Any]:
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"Registry '{key}' not found.") from None

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __copy__(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        new_registry = self.__class__()
        new_registry.update(self)
        return new_registry

    def copy(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        return self.__copy__()

    def get_registry(self, namespace: str) -> Registry[Any]:
        """Returns the registry for the given namespace."""
        if namespace not in self.__namespaces__:
            msg = f"Unknown registration namespace {namespace!r}."
            raise ValueError(msg)
        return getattr(self, namespace)

    @property
    def declarator_registries(self) -> dict[str, DeclaratorRegistry[Declarator]]:
        """Returns a mapping of namespace to declarator registries."""
        registries: dict[str, DeclaratorRegistry] = {}
        for namespace in self.__namespaces__:
            registry = self.get_registry(namespace)
            if isinstance(registry, DeclaratorRegistry):
                registries[namespace] = registry
        return registries

    @property
    def binding_registries(self) -> dict[str, BindingRegistry[Declarator]]:
        """Returns a mapping of namespace to binding registries."""
        registries: dict[str, BindingRegistry] = {}
        for namespace in self.__namespaces__:
            registry = self.get_registry(namespace)
            if isinstance(registry, BindingRegistry):
                registries[namespace] = registry
        return registries

    @abstractmethod
    def register(self, key: str, item: Any, *, namespace: str | None = None) -> None:
        if namespace is not None:
            registry = self.get_registry(namespace)
            registry.register(key, item)
            return
        raise NotImplementedError

    def update(self, other: Mapping[str, Registry[Any]]) -> None:
        """Update the multi-registry with items from another mapping."""
        declarators: dict[str, DeclaratorRegistry] = {}
        bindings: dict[str, BindingRegistry] = {}
        for namespace, registry in other.items():
            if namespace not in self.__namespaces__:
                msg = f"Unknown registration namespace {namespace!r}."
                raise ValueError(msg)
            elif isinstance(registry, DeclaratorRegistry):
                declarators[namespace] = registry
            elif isinstance(registry, BindingRegistry):
                bindings[namespace] = registry
            else:
                msg = f"Invalid registry type for namespace {namespace!r}."
                raise TypeError(msg)
        # Declarators get updated first, since bindings may depend on them.
        for namespace, registry in self.declarator_registries.items():
            declarator_updates = declarators.get(namespace, {})
            registry.update(declarator_updates)
        # In the case of bindings, even extant bindings must be updated,
        # since they may refer to new declarators.
        for namespace, registry in self.binding_registries.items():
            self_bindings = registry.to_dict()
            binding_updates = bindings.get(namespace, {})
            self_bindings.update(binding_updates)
            new_registry = BindingRegistry(_declarators=registry.declarators, **self_bindings)
            self._data[namespace] = new_registry

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to a plain nested dict suitable for serialization."""
        return {namespace: dict(registry) for namespace, registry in self._data.items() if registry}  # noqa

    def to_yaml(self) -> str:
        """YAML-style pretty print."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def __str__(self) -> str:
        return self.to_yaml()

# endregion
