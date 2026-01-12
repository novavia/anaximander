"""Defines the Declarator class and declarative metaclass.

These artifacts form the foundation for declarative type definitions in AML.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import ast
import re
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from functools import partial
from itertools import chain, count
from types import MappingProxyType
from typing import Any, ClassVar, Optional, Protocol, Self, TypedDict, TypeVar, get_args

import yaml
from attrs import define, field

from ..utils.meta import classproperty

# endregion

# =============================================================================
# Constants
# =============================================================================
# region Constants


declarator = partial(define, slots=False, frozen=True, kw_only=True)

DECLARATIVE_NAMESPACE: ContextVar[Optional["DeclarativeNamespace"]] = ContextVar("DECLARATIVE_NAMESPACE", default=None)  # noqa

type Assignment = ast.Assign | ast.AnnAssign
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
    working in conjunction with the declarative metaclass to process and register these declarations.
    """

    # Reserved names that cannot be used for protodescriptors.
    # These can be either strings or compiled regex patterns.
    # Strings restrict exact matches, while regex patterns allow for more complex rules.
    __reserved_patterns__: ClassVar[set[str | re.Pattern[str]]] = {
        re.compile(r"^__.*"),
    }

    __handles__: ClassVar[dict[str, type]] = {}  # Mapping of handles to declarator types (do not override) # noqa
    __handle__: ClassVar[Optional[str]]  = None # A plain-text lowercase handle for this declarator type (override in subclasses) # noqa
    _handles: ClassVar[tuple[str, ...]] = ()  # A tuple of all handles for this declarator type and its ancestors, in reverse mro (do not override) # noqa

    # Post-init wired fields (logically immutable; set via internal backdoor).
    name: str = field(init=False, default=None)  # Attribute or key name this declarator is assigned to # noqa
    owner: "declarative" = field(init=False, default=None) # Owning class of this declarator
    ordinal: int = field(init=False, default=None)  # Index of this declarator within the owning class # noqa
    __ast__: ast.AST = field(init=False, default=None)  # AST node that declared this declarator

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
    def handle(cls) -> Optional[str]:
        """The plain-text lowercase handle for this declarator type."""
        return cls.__handle__

    @classproperty
    def handles(cls) -> tuple[str, ...]:
        """A tuple of all handles for this declarator type and its ancestors."""
        return cls._handles

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
        if cls.__handle__ is not None:
            if cls.__handle__ in cls.__handles__:
                if cls.__handles__[cls.__handle__] not in cls.mro():
                    raise ValueError(
                        f"Declarator handle '{cls.__handle__}' is already registered."
                    )
            cls.__handles__[cls.__handle__] = cls
        handles = []
        for base in reversed(cls.mro()):
            if issubclass(base, Declarator) and base.__handle__ is not None:
                handles.append(base.__handle__)
        cls._handles = tuple(handles)

    def _set_once(self, attr: str, value: Any) -> None:
        """Internal backdoor: set a frozen attribute once (or idempotently)."""
        current = getattr(self, attr)
        if current is not None and current != value:
            raise RuntimeError(f"{self.__class__.__name__}.{attr} is already set.")
        object.__setattr__(self, attr, value)

    def __set_name__(self, owner: type, name: str):
        """Attach the name and owner class to this protodescriptor."""
        forbidden_names = {
            pattern for pattern in self.__reserved_patterns__ if isinstance(pattern, str)
        }
        forbidden_patterns = {
            pattern for pattern in self.__reserved_patterns__ if isinstance(pattern, re.Pattern)
        }
        if name in forbidden_names:
            mdtype = self.__class__.__name__
            msg = f"Cannot use reserved name {name} for Protodescriptor or type {mdtype}."
            raise ValueError(msg)
        if any(pattern.fullmatch(name) for pattern in forbidden_patterns):
            mdtype = self.__class__.__name__
            msg = f"Cannot use reserved name {name} for Protodescriptor or type {mdtype}."
            raise ValueError(msg)
        self._set_once("name", name)
        self._set_once("owner", owner)

    def __set_ast__(self, node: ast.AST | None) -> None:
        """Attach the AST node that declared this protodescriptor (if any)."""
        self._set_once("__ast__", node)

    def __validate__(self) -> None:
        """Validate this declarator after it has been bound to a class.

        This method is called by the declarative metaclass after the declarator
        has been assigned to a class attribute.
        """
        pass

    @abstractmethod
    def __bind__(self, value: Any, previous: Any) -> None:
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
        raise AttributeError(f"Declarator of type {self.__class__.__name__} cannot be overridden.")

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


class DeclarativeNamespace(dict):
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

    def register_declaration(self, declarator: Declarator) -> None:
        """Register a declarator in this namespace."""
        declarations: dict[int, Declarator] = self["__declarations__"]
        ordinal = next(self.declaration_index)
        declarations[ordinal] = declarator
        declarator._set_once("ordinal", ordinal)

    def register_binding(self, key: str, value: Any) -> None:
        """Register a binding in this namespace."""
        bindings: dict[str, Any] = self["__bindings__"]
        if key in bindings:
            raise KeyError(f"Binding '{key}' is already registered in this namespace.")
        bindings[key] = value

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
        self.bindings = BindingRegistry(_declarators=self)

    def register(self, key: str, item: D) -> None:
        """Registers a declarator."""
        if not isinstance(item, self.__types__):
            type_names = ", ".join(t.__name__ for t in self.__types__)
            raise TypeError(f"Declarator must be one of types: {type_names}")
        if key in self._data:
            if self.bindings is not None and key in self.bindings:
                raise RuntimeError(f"Cannot override declarator '{key}' with active binding.")
            try:
                self._data[key].__override__(item)
            except AttributeError as e:
                raise RuntimeError(f"Cannot override declarator '{key}': {e}") from e
        self._data[key] = item

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
        super().__init__(data, **kwargs)
        self._declarators_ref = (weakref.ref(_declarators))

    @property
    def declarators(self) -> DeclaratorRegistry[D] | None:
        """Return the referenced declarator registry, or None if it has been garbage collected."""
        return self._declarators_ref()

    def register(self, key: str, item: Any) -> None:
        """Register a binding."""
        declarators = self.declarators
        if declarators is None:
            raise RuntimeError("Cannot register binding: declarator registry has been garbage collected.")  # noqa
        try:
            declarator: D = declarators[key]
        except KeyError:
            raise KeyError(f"No declarator found for binding '{key}'.")
        try:
            extant_binding = self._data[key]
        except KeyError:
            self._data[key] = item
        else:
            try:
                declarator.__bind__(item, extant_binding)
            except AttributeError as e:
                raise RuntimeError(f"Cannot bind value to declarator '{key}': {e}") from e
            else:
                self._data[key] = item


class MultiRegistry(Mapping[str, Registry[Any]]):
    """A mapping that organizes multiple registries by namespace or rubric."""

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
        return self.__class__(self._data)

    def copy(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        return self.__copy__()

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to a plain nested dict suitable for serialization."""
        return {namespace: dict(registry) for namespace, registry in self._data.items() if registry}

    def to_yaml(self) -> str:
        """YAML-style pretty print."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def __str__(self) -> str:
        return self.to_yaml()

# endregion
