"""This module provides metaprogramming utilities."""

from typing import Callable


class AutoDecoratedType(type):
    __decorator__: Callable

    def __new__(mcl, name, bases, namespace):  # type: ignore
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
        super().__init_subclass__(**kwargs)
        setattr(cls, "__decorator__", decorator)
