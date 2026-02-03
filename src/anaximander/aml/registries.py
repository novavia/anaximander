# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Provide registry implementations for AML declarators, bindings, and modules.

Registries are the in-memory catalog for declarators and their bound values.
They make declarative types queryable by name, handle, and namespace, and they
power resolution of AML types across modules and projects.

This module defines the base registry abstractions, declarator/binding registries,
and multi-registry coordination that the prototype system and YAML serialization
rely on.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import Any, ClassVar, Self, get_args

from ..utils.yaml import nx_yaml_dump
from .declarative import declarative
from .declarators import MISSING, DeclarativeTypeKey, Declarator

# endregion

# =============================================================================
# Base registry
# =============================================================================
# region Base registry


class Registry[T](Mapping[str, T]):
    """A base registry class for declarator and binding registries."""

    def __init__(
        self,
        _data: Mapping[str, T] | Iterable[tuple[str, T]] | None = None,
        /,
        **kwargs: T,
    ) -> None:
        """Initialize the registry with optional data.

        Args:
            _data: Optional mapping or iterable of key/value pairs.
            **kwargs: Additional items to seed into the registry.
        """
        # Normalize inputs into a mutable dict for registry mutation.
        self._data: dict[str, T] = dict(_data or {}, **kwargs)

    def __getitem__(self, key: str) -> T:
        """Return the item stored under a given key.

        Args:
            key: Key to resolve.

        Returns:
            The stored item.

        Raises:
            KeyError: If the key does not exist.
        """
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(f"Key {key} not found in registry.") from None

    def __iter__(self):
        """Iterate over registry keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of registered items."""
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
        """Register an item with the given key.

        Args:
            key: Registry key for the item.
            item: Item to register.
        """
        raise NotImplementedError

    def update(self, other: Mapping[str, T]) -> None:
        """Update the registry with items from another mapping.

        Args:
            other: Mapping of items to register.
        """
        for key, item in other.items():
            self.register(key, item)

    def to_dict(self, *, compact: bool = True) -> dict[str, T | dict[str, Any] | str]:
        """Convert to a plain dict suitable for serialization.

        Args:
            compact: Whether to omit nested Declarator serialization.

        Returns:
            A dict representation of the registry.
        """
        if compact:
            return dict(self._data)
        return {
            key: (value.to_dict() if isinstance(value, Declarator) else value)
            for key, value in self._data.items()
        }

    def to_yaml(self, *, compact: bool = True) -> str:
        """Return a YAML serialization of the registry.

        Args:
            compact: Whether to omit nested Declarator serialization.

        Returns:
            A YAML string representation.
        """
        return nx_yaml_dump(self.to_dict(compact=compact))

    def __str__(self) -> str:
        """Return a human-readable YAML view."""
        return self.to_yaml()

    def __repr__(self) -> str:
        """Return a concise registry representation."""
        return f"<{type(self).__name__}>"

# endregion

# =============================================================================
# Declarative type registry
# =============================================================================
# region Declarative type registry


class DeclarativeTypeRegistry[M: ModuleType, DT: declarative](Registry[M]):
    """Registry for declarative types and their owning modules.

    This base registry stores module instances and indexes declarative types
    by their keys and names. Concrete subclasses decide how to resolve a type
    from a module (e.g., via module attributes or cached prototype lists).
    """

    def __init__(
        self,
        _data: Mapping[str, M] | Iterable[tuple[str, M]] | None = None,
        /,
        **kwargs: M,
    ) -> None:
        """Initialize the registry with optional module mappings.

        Args:
            _data: Optional mapping or iterable of module items.
            **kwargs: Additional modules keyed by their registration name.
        """
        super().__init__(_data, **kwargs)
        # Map declarative type keys to their source modules for lookups.
        self.prototype_to_module: dict[DeclarativeTypeKey, M] = {}
        # Map unqualified names to declarative keys for ambiguity resolution.
        self.name_to_prototypes: dict[str, set[DeclarativeTypeKey]] = {}

    def register(self, key: str, item: M) -> None:
        """Register a module by its canonical key.

        Args:
            key: Registry key (e.g., project::module).
            item: Module instance to register.

        Raises:
            KeyError: If a different module is already registered under the key.
        """
        registered = self._data.get(key)
        if registered is None:
            self._data[key] = item
            return
        if registered is not item:
            raise KeyError(f"Module {key} is already registered.")

    def register_module(self, module: M, types: Iterable[DT]) -> None:
        """Register a module and its declarative types.

        Args:
            module: Module instance holding declarative types.
            types: Declarative types defined in the module.
        """
        project = self._resolve_project(types)
        module_key = f"{project}::{self._module_name(module)}"
        self.register(module_key, module)
        # Index each type for name-based lookup.
        for type_ in types:
            self.register_type(type_, module)

    def register_type(self, type_: DT, module: M) -> None:
        """Register a declarative type for name-based lookup.

        Args:
            type_: Declarative type to register.
            module: Module that owns the type.

        Raises:
            TypeError: If the type does not define a declarative key.
        """
        if not hasattr(type_, "__key__"):
            raise TypeError(f"Type {type_} does not provide a declarative key.")
        key = type_.__key__
        # Track for both full-key and name-level lookups.
        self.prototype_to_module[key] = module
        self.name_to_prototypes.setdefault(key.name, set()).add(key)

    def resolve_type(self, name: str, *, project: str | None = None, module: str | None = None) -> DT:  # noqa
        """Resolve a declarative type name with optional project/module qualifiers.

        Args:
            name: Declarative type name.
            project: Optional project namespace qualifier.
            module: Optional module qualifier.

        Returns:
            The resolved declarative type.

        Raises:
            KeyError: If the type is unknown or ambiguous.
        """
        if project is not None and module is not None:
            key = DeclarativeTypeKey(project=project, module=module, name=name)
            try:
                return self._resolve_from_module(self.prototype_to_module[key], name)
            except KeyError:
                raise KeyError(f"Unknown type '{project}::{module}::{name}'.") from None
        keys = self.name_to_prototypes.get(name, set())
        if project is not None:
            keys = {k for k in keys if k.project == project}
        if module is not None:
            keys = {k for k in keys if k.module == module}
        if not keys:
            raise KeyError(f"Unknown type '{name}'.")
        if len(keys) > 1:
            candidates = ", ".join(f"{k.project}::{k.module}::{k.name}" for k in sorted(keys))
            raise KeyError(f"Ambiguous type '{name}': {candidates}")
        key = next(iter(keys))
        return self._resolve_from_module(self.prototype_to_module[key], key.name)

    def _module_name(self, module: M) -> str:
        """Return the canonical module name for registry keys."""
        return module.__name__

    def _resolve_from_module(self, module: M, name: str) -> DT:
        """Resolve a type name from a module instance.

        Args:
            module: Module to resolve against.
            name: Declarative type name.

        Returns:
            The resolved declarative type.
        """
        raise NotImplementedError

    def _resolve_project(self, types: Iterable[DT]) -> str:
        """Resolve the project namespace for a module's declarative types.

        Args:
            types: Declarative types from a module.

        Returns:
            The resolved project namespace.

        Raises:
            ValueError: If multiple projects are declared in the same module.
        """
        projects: set[str] = {
            p for type_ in types if (p := getattr(type_, "__project__", None)) is not None
        }
        if not projects:
            return "anaximander"
        if len(projects) > 1:
            msg = f"Module declares multiple projects: {sorted(projects)}"
            raise ValueError(msg)
        return next(iter(projects))

# endregion

# =============================================================================
# Declarator registries
# =============================================================================
# region Declarator registries


class DeclaratorRegistry[D: Declarator](Registry[D]):
    """A specialized registry for declarators."""

    __types__: ClassVar[tuple[type[Declarator], ...]] = (Declarator,)  # Admissible declarator types # noqa

    def __init_subclass__(cls):
        """Assume the first base is the base declarator registry."""
        super().__init_subclass__()
        # Resolve type specialization metadata for the subclass.
        base: type[DeclaratorRegistry] = cls.__bases__[0]
        super_types = getattr(base, "__types__", (Declarator,))
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
                raise TypeError(
                    "All types in __types__ must be subclasses of Declarator and "
                    "also subclasses of the base registry's __types__."
                )

    def __init__(
        self,
        _data: Mapping[str, D] | Iterable[tuple[str, D]] | None = None,
        /,
        **kwargs: D,
    ) -> None:
        """Initialize the declarator registry with optional items.

        Args:
            _data: Optional mapping or iterable of declarators.
            **kwargs: Additional declarators keyed by name.
        """
        super().__init__(_data, **kwargs)

    def __copy__(self) -> Self:
        """Create a shallow copy of the registry."""
        new_registry = self.__class__()
        new_registry._data = self._data.copy()
        return new_registry

    def register(self, key: str, item: D) -> None:
        """Register a declarator.

        Args:
            key: Registry key for the declarator.
            item: Declarator instance to register.

        Raises:
            TypeError: If the item is not an allowed declarator type.
            RuntimeError: If the declarator cannot be overridden.
        """
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
        """Return declarators matching the given types or handles.

        Args:
            *args: Declarator classes or handle strings.

        Returns:
            A registry containing matching declarators.

        Raises:
            ValueError: If a handle is unknown.
            TypeError: If provided types are not valid for this registry.
        """
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

# endregion

# =============================================================================
# Binding registries
# =============================================================================
# region Binding registries


class BindingRegistry[D: Declarator](Registry[Any]):
    """Registry for bindings."""

    def __init__(
        self,
        _data: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        /,
        *,
        _declarators: DeclaratorRegistry[D],
        **kwargs: Any,
    ) -> None:
        """Initialize the binding registry with an optional declarator registry.

        Args:
            _data: Optional mapping or iterable of bindings.
            _declarators: Declarator registry that validates bindings.
            **kwargs: Additional bindings keyed by name.

        Raises:
            ValueError: If declarators are passed as a keyword argument.
        """
        if "_declarators" in kwargs:
            raise ValueError("Declarators cannot be passed as a keyword argument.")
        self._declarators = _declarators
        # Register each binding so declarators can validate them.
        bindings = dict(_data or {}, **kwargs)
        self._data: dict[str, Any] = {}
        for key, item in bindings.items():
            self.register(key, item)

    @property
    def declarators(self) -> DeclaratorRegistry[D] :
        """Return the referenced declarator registry.

        Returns:
            The declarator registry used for validation.
        """
        return self._declarators

    def __copy__(self) -> Self:
        """Create a shallow copy of the binding registry."""
        new_registry = self.__class__(_declarators=self.declarators)
        new_registry._data = self._data.copy()
        return new_registry

    def register(self, key: str, item: Any) -> None:
        """Register a binding.

        Args:
            key: Binding name.
            item: Value to bind.

        Raises:
            RuntimeError: If the declarator registry is unavailable.
            KeyError: If no matching declarator exists.
            RuntimeError: If the binding cannot be applied.
        """
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

# endregion

# =============================================================================
# Multi-registry base
# =============================================================================
# region Multi-registry base


class MultiRegistry[R: Registry](Mapping[str, R]):
    """A mapping that organizes multiple registries by handle."""

    __handles__: ClassVar[set[str]] = set()  # Set of registry handles
    __namespaces__: ClassVar[set[str]] = set()  # Set of registry namespaces
    _registries: dict[str, R]  # Mapping of handle to registry
    _namespaces: dict[str, dict[str, str]]  # Mapping of namespace to name-to-handle mapping

    def __getitem__(self, handle: str) -> R:
        """Return the registry mapped to the handle.

        Args:
            handle: Registry handle key.

        Returns:
            The registry for the handle.

        Raises:
            KeyError: If no registry exists for the handle.
        """
        try:
            return self._registries[handle]
        except KeyError:
            raise KeyError(f"Registry '{handle}' not found.") from None

    def __iter__(self):
        """Iterate over registry handles."""
        return iter(self._registries)

    def __len__(self) -> int:
        """Return the number of registries."""
        return len(self._registries)

    def __copy__(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        new_registry = self.__class__()
        new_registry.update(self)
        return new_registry

    def copy(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        return self.__copy__()

    def get_registry(self, handle: str) -> R:
        """Return the registry for the given handle.

        Args:
            handle: Registry handle key.

        Returns:
            The registry registered under the handle.

        Raises:
            ValueError: If the handle is unknown.
        """
        if handle not in self.__handles__:
            msg = f"Unknown registration handle {handle!r}."
            raise ValueError(msg)
        return self._registries[handle]

    @classmethod
    def handle(cls, declarator: Declarator) -> str | None:
        """Map declarators to handles.

        Args:
            declarator: Declarator instance to map.

        Returns:
            The handle name or None.
        """
        return declarator.handle

    def _register_name(self, namespace: str, name: str, handle: str) -> None:
        """Register a name in the given namespace, mapped to handle.

        Args:
            namespace: Namespace label to register within.
            name: Name to register.
            handle: Registry handle that owns the name.

        Raises:
            ValueError: If the namespace is unknown.
            KeyError: If the name is already registered under another handle.
        """
        if namespace not in self.__namespaces__:
            msg = f"Unknown registration namespace {namespace!r}."
            raise ValueError(msg)
        names = self._namespaces[namespace]
        registered_handle = names.get(name)
        if registered_handle is None:
            names[name] = handle
            return
        elif registered_handle == handle:
            return
        else:
            msg = f"""Cannot register {name!r} with handle {handle!r} in namespace {namespace!r}
                because it is already registered with {registered_handle!r}."""
            raise KeyError(msg)

    @abstractmethod
    def register(self, name: str, item: Any, *, handle: str | None = None, namespace: str | None = None) -> None:  # noqa
        """Register an item in the appropriate registry.

        Args:
            name: Name to register.
            item: Item to register.
            handle: Optional explicit handle.
            namespace: Optional explicit namespace.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, other: "MultiRegistry[R]") -> None:
        """Update the multi-registry with items from another mapping.

        Args:
            other: Multi-registry to merge.

        Raises:
            TypeError: If the registry types do not match.
        """
        if type(other) is not type(self):
            msg = "Can only merge with another MultiRegistry instance of the same type."
            raise TypeError(msg)

    def to_dict(self, *, compact: bool = True) -> dict[str, dict[str, Any]]:
        """Convert to a plain nested dict suitable for serialization.

        Args:
            compact: Whether to omit nested Declarator serialization.

        Returns:
            A nested dict representation.
        """
        return {
            handle: registry.to_dict(compact=compact)
            for handle, registry in self._registries.items()
            if registry
        }

    def to_yaml(self, *, compact: bool = True) -> str:
        """Return a YAML serialization of the multi-registry.

        Args:
            compact: Whether to omit nested Declarator serialization.

        Returns:
            A YAML string representation.
        """
        return nx_yaml_dump(self.to_dict(compact=compact))

    def __str__(self) -> str:
        """Return a human-readable YAML view."""
        return self.to_yaml()

    def __repr__(self) -> str:
        """Return a concise multi-registry representation."""
        return f"<{type(self).__name__}>"

# endregion

# =============================================================================
# Multi declarator registry
# =============================================================================
# region Multi declarator registry


class MultiDeclaratorRegistry(MultiRegistry[DeclaratorRegistry]):
    """A multi-registry specialized for declarators."""

    @classmethod
    def namespace(cls, declarator: Declarator) -> str | None:
        """Map declarators to namespaces.

        Args:
            declarator: Declarator instance to map.

        Returns:
            The namespace label or None.
        """
        return None

    def _register_declarator(self, name: str, declarator: Declarator, *, handle: str | None = None, namespace: str | None = None) -> None:  # noqa
        """Register a declarator in the appropriate registry and namespace.

        Args:
            name: Declarator name.
            declarator: Declarator instance.
            handle: Optional explicit handle.
            namespace: Optional explicit namespace.

        Raises:
            TypeError: If the handle cannot be determined.
        """
        if handle is None:
            handle = self.handle(declarator)
            if handle is None:
                msg = f"Cannot determine registration handle for item of type {type(declarator).__name__}."  # noqa
                raise TypeError(msg)
        # Check that the handle corresponds to a declarator registry
        registry = self.get_registry(handle)
        # Prevent name clashes across registries
        if namespace is None:
            namespace = self.namespace(declarator)
        if namespace is not None:
            self._register_name(namespace, name, handle)
        registry.register(name, declarator)

    def register(self, name: str, item: Declarator, *, handle: str | None = None, namespace: str | None = None) -> None:  # noqa
        """Register a declarator in the appropriate registry.

        Args:
            name: Declarator name.
            item: Declarator instance.
            handle: Optional explicit handle.
            namespace: Optional explicit namespace.

        Raises:
            TypeError: If the item is not a declarator.
        """
        if not isinstance(item, Declarator):
            raise TypeError("Only Declarator instances can be registered in MultiDeclaratorRegistry.")  # noqa
        self._register_declarator(name, item, handle=handle, namespace=namespace)

    def update(self, other: "MultiRegistry[DeclaratorRegistry]") -> None:
        """Update the multi-registry with items from another instance.

        Args:
            other: Multi-registry to merge.

        Raises:
            TypeError: If registry types do not match.
        """
        if type(other) is not type(self):
            msg = "Can only merge with another MultiDeclaratorRegistry instance of the same type."
            raise TypeError(msg)
        # Declarators get updated first, since bindings may depend on them.
        for handle in self.__handles__:
            declarator_updates = other.get_registry(handle)
            for name, declarator in declarator_updates.items():
                self._register_declarator(name, declarator, handle=handle)

# endregion

# =============================================================================
# Multi binding registry
# =============================================================================
# region Multi binding registry


class MultiBindingRegistry(MultiRegistry[BindingRegistry]):
    """A multi-registry specialized for bindings."""

    _declarators: MultiDeclaratorRegistry  # Referenced declarator multi-registry
    __auto_handles__: ClassVar[set[str]] = set()  # Handles that can be auto-detected

    def __init__(self, declarators: MultiDeclaratorRegistry) -> None:
        """Initialize the multi-binding registry with a declarator multi-registry.

        Args:
            declarators: Declarator registry used for binding validation.
        """
        super().__init__()
        self._declarators = declarators

    @property
    def declarators(self) -> MultiDeclaratorRegistry:
        """Return the referenced declarator multi-registry.

        Returns:
            The declarator multi-registry.
        """
        return self._declarators

    def __copy__(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        new_registry = self.__class__(self._declarators)
        new_registry.update(self)
        return new_registry

    def register(self, name: str, item: Any, *, handle: str | None = None, namespace: str | None = None) -> None:  # noqa
        """Register a binding in the appropriate registry.

        Args:
            name: Binding name.
            item: Value to bind.
            handle: Optional explicit handle.
            namespace: Optional explicit namespace.

        Raises:
            KeyError: If no matching declarator registry accepts the binding.
        """
        if handle is not None:
            registry = self.get_registry(handle)
            registry.register(name, item)
            return
        for handle in self.__auto_handles__:
            registry = self.get_registry(handle)
            try:
                registry.register(name, item)
                return
            except KeyError:
                continue
        msg = f"Cannot register binding '{name}': no matching declarator found."
        raise KeyError(msg)

    def update(self, other: "MultiRegistry[BindingRegistry]", *, update_declarators: bool = False) -> None:  # noqa
        """Update the multi-registry with items from another mapping.

        Args:
            other: Multi-registry to merge.
            update_declarators: Whether to merge declarators as well.

        Raises:
            TypeError: If registry types do not match.
        """
        if type(other) is not type(self):
            msg = "Can only merge with another MultiBindingRegistry instance of the same type."
            raise TypeError(msg)
        # In the case of bindings, even extant bindings must be updated,
        # since they may refer to new declarators.
        for handle in self.__handles__:
            extant_bindings = self.get_registry(handle)
            extant_declarators = extant_bindings.declarators
            binding_updates = other.get_registry(handle)
            declarator_updates = binding_updates.declarators
            if update_declarators:
                extant_declarators.update(declarator_updates)
            # Merge bindings
            all_bindings = extant_bindings.to_dict() | binding_updates.to_dict()
            new_registry = BindingRegistry(_declarators=extant_declarators, **all_bindings)
            self._registries[handle] = new_registry

# endregion
