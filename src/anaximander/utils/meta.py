"""Metaprogramming utilities for Anaximander."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from typing import Any, Callable

# endregion

# =============================================================================
# Metaclasses and descriptors
# =============================================================================
# region Metaclasses and descriptors


class AutoDecoratedType(type):
    """Metaclass that automatically decorates newly created classes.

    A decorator is configured during subclass initialization and applied at class
    creation time unless the class is explicitly marked as already decorated.
    """

    __decorator__: Callable

    def __new__(mcls, name, bases, namespace) -> type:
        """Create a class and apply the configured decorator if not marked decorated.

        Args:
            mcls (type): The metaclass.
            name (str): Name of the class being created.
            bases (tuple[type, ...]): Base classes.
            namespace (dict): Class namespace.

        Returns:
            type: The resulting class, possibly decorated.
        """
        new_class = super().__new__(mcls, name, bases, namespace)
        if "__decorated__" in namespace:
            delattr(new_class, "__decorated__")
            return new_class
        setattr(new_class, "__decorated__", True)
        decorated = mcls.__decorator__(new_class, auto_attribs=True)
        return decorated

    def __init_subclass__(cls, /, decorator: Callable, **kwargs) -> None:
        """Configure the decorator used to auto-decorate subclasses.

        Args:
            decorator (Callable): Decorator to apply to subsequently created subclasses.
            **kwargs: Additional keyword arguments forwarded to the superclass.
        """
        super().__init_subclass__(**kwargs)
        setattr(cls, "__decorator__", decorator)


class classproperty[T]:
    """A descriptor that behaves like a property for both classes and instances.

    Unlike standard properties or classmethods, this descriptor ensures the getter
    receives the class (owner) as its first argument regardless of whether it is
    accessed via the class itself or one of its instances.
    """

    # Use a permissive callable type so methods annotated with a concrete
    # class (e.g. def x(cls) -> int) are accepted by type checkers.
    def __init__(self, fget: Callable[..., T]) -> None:
        self.fget: Callable[..., T] = fget

    def __get__(self, instance: Any, owner: type) -> T:
        # owner is the class (e.g., DataDescriptor)
        # instance is the instance if accessed via instance, or None if via class
        return self.fget(owner)


class Singleton(type):
    """A metaclass for singleton classes."""
    _instances: dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# endregion
