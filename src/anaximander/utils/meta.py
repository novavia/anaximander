"""This module provides metaprogramming utilities."""

from typing import Callable


class AutoDecoratedType(type):
    """Metaclass that automatically decorates newly created classes.

    A decorator is configured during subclass initialization and applied at class
    creation time unless the class is explicitly marked as already decorated.
    """

    __decorator__: Callable

    def __new__(mcl, name, bases, namespace):  # type: ignore
        """Create a class and apply the configured decorator if not marked decorated.

        Args:
            mcl (type): The metaclass.
            name (str): Name of the class being created.
            bases (tuple[type, ...]): Base classes.
            namespace (dict): Class namespace.

        Returns:
            type: The resulting class, possibly decorated.
        """
        new_class = super().__new__(mcl, name, bases, namespace)
        if "__decorated__" in namespace:
            delattr(new_class, "__decorated__")
            return new_class
        setattr(new_class, "__decorated__", True)
        decorated = mcl.__decorator__(new_class, auto_attribs=True)
        return decorated

    # def __init__(cls, name, bases, namespace, decorator: Callable):  # type: ignore
    #     super().__init__(name, bases, namespace)

    def __init_subclass__(cls, /, decorator: Callable, **kwargs) -> None:
        """Configure the decorator used to auto-decorate subclasses.

        Args:
            decorator (Callable): Decorator to apply to subsequently created subclasses.
            **kwargs: Additional keyword arguments forwarded to the superclass.
        """
        super().__init_subclass__(**kwargs)
        setattr(cls, "__decorator__", decorator)
