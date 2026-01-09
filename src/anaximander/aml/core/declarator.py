"""Defines the Declarator class, which is responsible for declaring descriptors, methods and metadata in the AML language."""

import ast
import datetime
import re
from abc import ABC, abstractmethod
from contextvars import ContextVar, Token
from itertools import chain, count
from types import MappingProxyType
from typing import (
    ChainMap,
    ClassVar,
    Collection,
    Optional,
    Protocol,
    Mapping,
    Any,
    TypedDict
)
import weakref

from attrs import define, field


# Sentinel value for unspecified defaults
class _MissingSentinel:
    """Unique sentinel for unspecified defaults."""

    def __repr__(self) -> str:
        return "MISSING"


MISSING: _MissingSentinel = _MissingSentinel()


def _is_temporal(type_: Any) -> bool:
    """Return True when a type behaves like a timestamp or date."""
    if not isinstance(type_, type):
        return False
    if issubclass(type_, (datetime.datetime, datetime.date, datetime.time)):
        return True
    return bool(getattr(type_, "__time_like__", False) or getattr(type_, "__temporal__", False))


def _is_spatial(type_: Any) -> bool:
    """Return True when a type represents a geometry/location value."""
    if not isinstance(type_, type):
        return False
    return bool(
        getattr(type_, "__geometry__", False)
        or getattr(type_, "__geo__", False)
        or getattr(type_, "__geom__", False)
    )


DECLARATIVE_NAMESPACE: ContextVar[Optional["DeclarativeNamespace"]] = ContextVar("DECLARATIVE_NAMESPACE", default=None)  # noqa


type Assignment = ast.Assign | ast.AnnAssign


class DeclarativeTypeKey(TypedDict):
    """A unique key for identifying declarative types."""
    project: str
    module: str
    name: str


class DeclaratorKey(TypedDict):
    """A unique key for identifying declarators."""
    owner: DeclarativeTypeKey
    index: int


class DeclaratorConfig(Protocol):
    """A protocol for extending declarator configuration."""

    def resolve(self, **context) -> Mapping[str, Any]: ...


type ConfigValue = Any | DeclaratorConfig | Mapping[str, ConfigValue]
type Config = DeclaratorConfig | Mapping[str, ConfigValue]



@define
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

    __handle__: ClassVar[Optional[str]]  # A plain-text lowercase handle for this declarator type (override in subclasses) # noqa
    __handles__: ClassVar[dict[str, type]] = {}  # Mapping of handles to declarator types (do not override) # noqa

    # Post-init wired fields (logically immutable; set via internal backdoor).
    name: str = field(init=False, default=None)  # Attribute or key name this declarator is assigned to # noqa
    owner: "declarative" = field(init=False, default=None) # Owning class of this declarator
    index: int = field(init=False, default=None)  # Index of this declarator within the owning class # noqa
    __ast__: ast.AST = field(init=False, default=None)  # AST node that declared this declarator

    # Init-time fields (immutable)
    doc: str | None = field(default=None)  # Optional documentation string
    config: Mapping[str, ConfigValue] = field(factory=dict)  # Extraneous declarator configuration

    @property
    def __key__(self) -> DeclaratorKey:
        """A unique key for identifying this declarator."""
        if self.owner is None or self.index is None:
            raise RuntimeError("Declarator must be bound to a class before accessing its key.")
        return DeclaratorKey(
            owner=self.owner.__key__,
            index=self.index,
        )

    def __attrs_post_init__(self) -> None:
        """Post-initialization processing for the declarator."""
        dns = DECLARATIVE_NAMESPACE.get()
        if dns is not None:
            dns.register_declaration(self)
        # Freeze config to prevent accidental mutation.
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    def __init_subclass__(cls):
        super().__init_subclass__()
        parent = cls.mro()[1]
        reserved_patterns: set[str | re.Pattern[str]] = getattr(
            parent, "__reserved_patterns__", set()
        )
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


# class Registry[str, T](Mapping[str, T]):
#     """A base registry class for declarators and bindings.

#     The registration structure is a chain-mapped dictionary which keeps track of inheritance and
#     allows for easy lookup and iteration over registered items.
#     """

#     def __init__(self, *, parent: "Registry[str, T] | None" = None):
#         if parent is not None:
#             self._data: ChainMap[str, T] = ChainMap({}, *parent._data.maps)
#         else:
#             self._data: ChainMap[str, T] = ChainMap()
#         self.parent = parent

#     def __getitem__(self, key: str) -> T:
#         try:
#             return self._data[key]
#         except KeyError:
#             raise KeyError(f"Key {key} not found in registry.") from None

#     def __iter__(self):
#         return iter(self._data)

#     def __len__(self) -> int:
#         return len(self._data)

#     def __copy__(self) -> "Registry[str, T]":
#         """Create a shallow copy of the registry."""
#         new_registry = self.__class__()
#         new_registry._data = self._data.copy()
#         return new_registry

#     def copy(self) -> "Registry[str, T]":
#         """Create a shallow copy of the registry."""
#         return self.__copy__()

#     @abstractmethod
#     def register(self, key: str, item: T) -> None:
#         """Register an item with the given key."""
#         raise NotImplementedError

#     def update(self, other: Mapping[str, T]) -> None:
#         """Update the registry with items from another mapping."""
#         for key, item in other.items():
#             self.register(key, item)

#     @abstractmethod
#     def filter(self, *, recursive: bool = True) -> Mapping[str, T]:
#         """Yield items that satisfy the given predicate.

#         recursive indicates whether to search parent registries.
#         """
#         if recursive:
#             data = self._data
#         else:
#             data = self._data.maps[0]
#         return {k: v for k, v in data.items()}


# class BindingRegistry(Registry[str, Any]):
#     """Registry for bindings within a class namespace."""

#     def __init__(self, declarator_registry: "DeclaratorRegistry"):
#         declarator_parent = declarator_registry.parent
#         # Use the parent's bindings only if the parent is a DeclaratorRegistry instance.
#         parent_bindings = (
#             declarator_parent.bindings
#             if isinstance(declarator_parent, DeclaratorRegistry)
#             else None
#         )
#         super().__init__(parent=parent_bindings)
#         self.declarators = weakref.ref(declarator_registry)  # Avoid circular reference

#     def register(self, key: str, item: Any) -> None:
#         """Register a binding."""
#         declarators = self.declarators()
#         if declarators is None:
#             raise RuntimeError("Declarator registry reference has been garbage collected.")
#         try:
#             declarator = declarators[key]
#         except KeyError:
#             raise KeyError(f"No declarator found for binding '{key}'.")
#         extant_binding = self._data.get(key, None)
#         try:
#             declarator.__bind__(item, extant_binding)
#         except AttributeError as e:
#             raise RuntimeError(f"Cannot bind value to declarator '{key}': {e}") from e
#         self._data[key] = item

#     def filter(self, *, recursive: bool = True) -> Mapping[str, Any]:
#         """Yield bindings that match the given types or handles."""
#         return super().filter(recursive=recursive)


# class DeclaratorRegistry(Registry[str, Declarator]):
#     """Registry for declarators within a class namespace."""

#     __types__: ClassVar[tuple[type[Declarator], ...]]  # Admissible declarator types

#     def __init__(
#         self,
#         *,
#         parent: "DeclaratorRegistry | None" = None,
#         declarator_container: str | None = None,
#         binding_container: str | None = None,
#     ):
#         if parent is not None:
#             if parent.__class__ is not self.__class__:
#                 raise TypeError("A registry must be of the same class as its parent.")
#         super().__init__(parent=parent)
#         self.declarator_container = declarator_container
#         self.binding_container = binding_container
#         self.bindings: BindingRegistry = BindingRegistry(self)

#     def register(self, key: str, item: Declarator) -> None:
#         """Register a declarator."""
#         if not isinstance(item, self.__types__):
#             type_names = ", ".join(t.__name__ for t in self.__types__)
#             raise TypeError(f"Declarator must be one of types: {type_names}")
#         if key in self._data:
#             try:
#                 self._data[key].__override__(item)
#             except AttributeError as e:
#                 raise RuntimeError(f"Cannot override declarator '{key}': {e}") from e
#         self._data[key] = item

#     def filter(self, *args: type | str, recursive: bool = True) -> Mapping[str, Declarator]:
#         """Yield declarators that match the given types or handles."""
#         data = super().filter(recursive=recursive)
#         if not args:
#             return data
#         types = []
#         for arg in args:
#             if isinstance(arg, type):
#                 types.append(arg)
#             elif isinstance(arg, str):
#                 try:
#                     types.append(Declarator.__handles__[arg])
#                 except KeyError:
#                     raise ValueError(f"Unknown declarator handle: {arg}") from None
#             else:
#                 raise TypeError("Arguments must be types or declarator handles (strings).")
#         if not all(any(issubclass(t, __t__) for t in types) for __t__ in self.__types__):
#             type_names = ", ".join(t.__name__ for t in self.__types__)
#             raise TypeError(f"Arguments must be one of types: {type_names} or their handles.")
#         return {k: v for k, v in data.items() if isinstance(v, tuple(types))}

#     def update(self, other: Mapping[str, Declarator]) -> None:
#         """Update the registry with declarators from another mapping."""
#         super().update(other)
#         try:
#             other_bindings = getattr(other, "bindings")
#             assert isinstance(other_bindings, BindingRegistry)
#         except (AttributeError, AssertionError):
#             pass
#         else:
#             self.bindings.update(other_bindings)


class DeclarativeNamespace(dict):
    """Collects class body declarations for a declarative type."""
    context_token: Token

    def __init__(self, *, strict: bool = True, bindable_names: list[str] | None = None):
        super().__init__(__declarations__={}, __bindings__={})
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
        index = next(self.declaration_index)
        declarations[index] = declarator
        declarator._set_once("index", index)

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
