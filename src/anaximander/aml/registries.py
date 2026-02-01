"""Defines the Declarator class and declarative metaclass.

These artifacts form the foundation for declarative type definitions in AML.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from abc import abstractmethod
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import (
    Any,
    ClassVar,
    Self,
    get_args,
)

from ..utils.yaml import nx_yaml_dump
from .declarative import declarative
from .declarators import MISSING, DeclarativeTypeKey, Declarator

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
        return nx_yaml_dump(self.to_dict())

    def __str__(self) -> str:
        return self.to_yaml()


class DeclarativeTypeRegistry[M: ModuleType, DT: declarative](Registry[M]):
    """Registry for declarative types and their owning modules.

    This base registry stores module instances and indexes declarative types
    by their keys and names. Concrete subclasses decide how to resolve a type
    from a module (e.g., via module attributes or cached prototype lists).
    """

    def __init__(self, data: Mapping[str, M] | Iterable[tuple[str, M]] | None = None, **kwargs: M) -> None:  # noqa
        """Initialize the registry with optional module mappings."""
        super().__init__(data, **kwargs)
        self.prototype_to_module: dict[DeclarativeTypeKey, M] = {}
        self.name_to_prototypes: dict[str, set[DeclarativeTypeKey]] = {}

    def register(self, key: str, item: M) -> None:
        """Register a module by its canonical key."""
        registered = self._data.get(key)
        if registered is None:
            self._data[key] = item
            return
        if registered is not item:
            raise KeyError(f"Module {key} is already registered.")

    def register_module(self, module: M, types: Iterable[DT]) -> None:
        """Register a module and its declarative types."""
        project = self._resolve_project(types)
        module_key = f"{project}::{self._module_name(module)}"
        self.register(module_key, module)
        for type_ in types:
            self.register_type(type_, module)

    def register_type(self, type_: DT, module: M) -> None:
        """Register a declarative type for name-based lookup."""
        if not hasattr(type_, "__key__"):
            raise TypeError(f"Type {type_} does not provide a declarative key.")
        key = type_.__key__
        self.prototype_to_module[key] = module
        self.name_to_prototypes.setdefault(key.name, set()).add(key)

    def resolve_type(self, name: str, *, project: str | None = None, module: str | None = None) -> DT:  # noqa
        """Resolve a declarative type name with optional project/module qualifiers."""
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
        """Resolve a type name from a module instance."""
        raise NotImplementedError

    def _resolve_project(self, types: Iterable[DT]) -> str:
        """Resolve the project namespace for a module's declarative types."""
        projects: set[str] = {p for type_ in types if (p := getattr(type_, "__project__", None)) is not None}  # noqa
        if not projects:
            return "anaximander"
        if len(projects) > 1:
            msg = f"Module declares multiple projects: {sorted(projects)}"
            raise ValueError(msg)
        return next(iter(projects))


class DeclaratorRegistry[D: Declarator](Registry[D]):
    """A specialized registry for declarators."""

    __types__: ClassVar[tuple[type[Declarator], ...]] = (Declarator,)  # Admissible declarator types # noqa

    def __init_subclass__(cls):
        """Assumes the first base is the base declarator registry."""
        super().__init_subclass__()
        base: type[DeclaratorRegistry]= cls.__bases__[0]
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


class MultiRegistry[R: Registry](Mapping[str, R]):
    """A mapping that organizes multiple registries by handle."""

    __handles__: ClassVar[set[str]] = set()  # Set of registry handles
    __namespaces__: ClassVar[set[str]] = set()  # Set of registry namespaces
    _registries: dict[str, R]  # Mapping of handle to registry
    _namespaces: dict[str, dict[str, str]]  # Mapping of namespace to name-to-handle mapping

    def __getitem__(self, handle: str) -> R:
        try:
            return self._registries[handle]
        except KeyError:
            raise KeyError(f"Registry '{handle}' not found.") from None

    def __iter__(self):
        return iter(self._registries)

    def __len__(self) -> int:
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
        """Returns the registry for the given handle."""
        if handle not in self.__handles__:
            msg = f"Unknown registration handle {handle!r}."
            raise ValueError(msg)
        return self._registries[handle]

    @classmethod
    def handle(cls, declarator: Declarator) -> str | None:
        """This customizable method maps declarators to handles."""
        return declarator.handle

    def _register_name(self, namespace: str, name: str, handle: str) -> None:
        """Registers a name in the given namespace, mapped to handle."""
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
        """Register an item in the appropriate registry."""
        raise NotImplementedError

    @abstractmethod
    def update(self, other: "MultiRegistry[R]") -> None:
        """Update the multi-registry with items from another mapping."""
        if type(other) is not type(self):
            msg = "Can only merge with another MultiRegistry instance of the same type."
            raise TypeError(msg)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to a plain nested dict suitable for serialization."""
        return {handle: dict(registry) for handle, registry in self._registries.items() if registry}  # noqa

    def to_yaml(self) -> str:
        """YAML-style pretty print."""
        return nx_yaml_dump(self.to_dict())

    def __str__(self) -> str:
        return self.to_yaml()


class MultiDeclaratorRegistry(MultiRegistry[DeclaratorRegistry]):
    """A multi-registry specialized for declarators."""

    @classmethod
    def namespace(cls, declarator: Declarator) -> str | None:
        """This customizable method maps declarators to namespaces."""
        return None

    def _register_declarator(self, name: str, declarator: Declarator, *, handle: str | None = None, namespace: str | None = None) -> None:  # noqa
        """Registers a declarator in the appropriate registry and namespace."""
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
        """Register a declarator in the appropriate registry."""
        if not isinstance(item, Declarator):
            raise TypeError("Only Declarator instances can be registered in MultiDeclaratorRegistry.")  # noqa
        self._register_declarator(name, item, handle=handle, namespace=namespace)

    def update(self, other: "MultiRegistry[DeclaratorRegistry]") -> None:
        """Update the multi-registry with items from another instance."""
        if type(other) is not type(self):
            msg = "Can only merge with another MultiDeclaratorRegistry instance of the same type."
            raise TypeError(msg)
        # Declarators get updated first, since bindings may depend on them.
        for handle in self.__handles__:
            declarator_updates = other.get_registry(handle)
            for name, declarator in declarator_updates.items():
                self._register_declarator(name, declarator, handle=handle)


class MultiBindingRegistry(MultiRegistry[BindingRegistry]):
    """A multi-registry specialized for bindings."""

    _declarators: MultiDeclaratorRegistry  # Referenced declarator multi-registry
    __auto_handles__: ClassVar[set[str]] = set()  # Handles that can be auto-detected

    def __init__(self, declarators: MultiDeclaratorRegistry) -> None:
        """Initialize the multi-binding registry with a declarator multi-registry."""
        super().__init__()
        self._declarators = declarators

    @property
    def declarators(self) -> MultiDeclaratorRegistry:
        """Return the referenced declarator multi-registry."""
        return self._declarators

    def __copy__(self) -> Self:
        """Create a shallow copy of the multi-registry."""
        new_registry = self.__class__(self._declarators)
        new_registry.update(self)
        return new_registry

    def register(self, name: str, item: Any, *, handle: str | None = None, namespace: str | None = None) -> None:  # noqa
        """Register a binding in the appropriate registry."""
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
        """Update the multi-registry with items from another mapping."""
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
