"""Defines the Declarator class and declarative metaclass.

These artifacts form the foundation for declarative type definitions in AML.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import ast
from abc import abstractmethod
from contextvars import ContextVar, Token
from itertools import chain, count
from typing import (
    Any,
    ClassVar,
    Optional,
)

from attrs import define

from .declarators import Declarator

# endregion

# =============================================================================
# Declarative metaclass
# =============================================================================
# region Declarative metaclass

# TODO: moved from declarators post-init - make an explict registration instead
        # dns = DECLARATIVE_NAMESPACE.get()
        # if dns is not None:
        #     dns.register_declarator(self)

# Context variable holding the current declarative namespace
DECLARATIVE_NAMESPACE: ContextVar[Optional["DeclarativeNamespace"]] = ContextVar("DECLARATIVE_NAMESPACE", default=None)  # noqa


@define(frozen=True)
class DeclarativeTypeKey:
    """A unique key for identifying declarative types."""
    project: str
    module: str
    name: str


class DeclarativeNamespace(dict[str, Any]):
    """Collects class body declarations for a declarative type."""
    context_token: Token

    def __init__(self, *, strict: bool = True, bindable_domain_names: set[str] | None = None):
        super().__init__(__raw_declarators__=dict(), __raw_bindings__=dict())
        self.strict = strict
        self.bindable_domain_names = bindable_domain_names or set()
        self.declaration_index = count(start=1)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self:
            raise RuntimeError(f"Cannot redefine name '{key}' in declarative namespace.")
        elif isinstance(value, Declarator):
            self.register_declarator(value, name=key)
        elif key in self.bindable_domain_names:
            self.register_binding(key, value)
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        raise RuntimeError("Cannot delete items from a declarative namespace.")

    def register_declarator(self, declarator: Declarator, *, name: str | None = None) -> None:
        """Registers a declarator in this namespace.

        name is optional and only used when the declarator is registered through a declarative
        interface with a .declare() method. Otherwise the declarator is not given a name yet as
        it is set by the __set_name__ hook after the class body is executed.
        """
        declarators: dict[int, Declarator] = self["__raw_declarators__"]
        if declarator in declarators.values():
            return
        if not isinstance(declarator, Declarator):
            raise TypeError(f"Expected a Declarator instance, got {declarator}.")
        if name is not None:
            # sentinel to fail quick if name contains a forbidden dot
            if "." in name:
                raise ValueError("Declarator names cannot contain '.'.")
            # next we check for name conflicts, starting with a cursory lookup
            # and then refining it at the namespace level if warranted
            if name in (d.name for d in declarators.values()):
                declarator_handle = declarator.handle
                matches = [d for d in declarators.values() if d.name == name]
                if any(d.handle == declarator_handle for d in matches):
                    msg = f"Duplicate declaration for name '{name}'"
                    if declarator_handle:
                        msg += f" and handle '{declarator_handle}'."
                    raise KeyError(msg)
            declarator._set_once("name", name, treat_none_as_unset=True)
        ordinal = next(self.declaration_index)
        declarators[ordinal] = declarator
        declarator._set_once("ordinal", ordinal, treat_none_as_unset=True)

    def register_binding(self, key: str, value: Any, *, handle: str | None= None) -> None:
        """Register a binding in this namespace.

        The handle parameter optionally specifies the handle under which the binding is registered.
        In that case, the binding key is prefixed with the handle and a dot. This allows bindings
        to be namespaced under different declarator types. The default is no prefix, and applies
        to bindings registered directly in the namespace.
        """
        bindings: dict[str, Any] = self["__raw_bindings__"]
        if handle is not None:
            binding_key = f"{handle}.{key}"
            if binding_key in bindings:
                raise KeyError(f"Binding '{key}' is already registered in with handle '{handle}'.")
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
                    if (
                        not key.startswith("__")
                        and key not in self["__raw_declarators__"]
                        and key not in self["__raw_bindings__"]
                    ):  # noqa
                        raise RuntimeError(
                            f"Name '{key}' is neither a declaration nor a binding in strict mode."
                        )
        finally:
            DECLARATIVE_NAMESPACE.reset(self.context_token)


class declarative(type):
    """Metaclass for declarative types."""

    __project__: ClassVar[str] = "anaximander"  # Project name for this declarative type
    __raw_declarators__: dict[int, Declarator]  # Declarators declared in this type
    __raw_bindings__: dict[str, Any]  # Bindings made in this type
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
    def _bindable_domain_names(cls) -> set[str]:
        """Names that can be bound in the body of this declarative type's subclasses."""
        return set()

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs) -> DeclarativeNamespace:
        """Collects declarations, assignments and containers in the class body."""
        declarative_parents = [b for b in bases if isinstance(b, declarative)]
        bindable_domain_names = set(
            chain(*(b._bindable_domain_names for b in declarative_parents))
        )
        namespace = DeclarativeNamespace(
            strict=mcls.__strict__,
            bindable_domain_names=bindable_domain_names,
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
        for declarator in cls.__raw_declarators__.values():
            if declarator.owner is None:
                name = getattr(declarator, "name", None)
                if name is None:
                    raise RuntimeError("Unnamed declarator registered outside class assignment.")
                declarator.__set_name__(cls, name)
        # Runs declarator validation hooks
        for declarator in cls.__raw_declarators__.values():
            declarator.__validate__()
        return cls

# endregion
