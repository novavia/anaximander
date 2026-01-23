"""Namespace objects for AML declarators and metadata bindings."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


from collections.abc import Mapping
from typing import Any, Callable

from anaximander.utils.meta import Singleton

from .declarative import (
    DECLARATIVE_NAMESPACE,
    MISSING,
    DeclarativeNamespace,
    Declarator,
    EnumerationCallableDeclarator,
    Missing,
)
from .metadescriptors import (
    MetadataDeclarator,
    MetadataValidator,
    NxFieldDeclarator,
    NxFieldValidator,
    OptionDeclarator,
    OptionValidator,
)

# endregion

# =============================================================================
# Namespace classes
# =============================================================================
# region Namespace classes


class DeclaratorNamespace(metaclass=Singleton):
    """Base class for AML declarator namespaces.

    Instances act as lightweight facades for creating declarators and registering
    declarations or bindings in the active declarative class body.
    """

    __declarator_type__: type[Declarator]
    __validator_type__: type[EnumerationCallableDeclarator]
    __namespace_name__: str

    @property
    def declarator_type(self) -> type[Declarator]:
        """The declarator type for this namespace."""
        return self.__class__.__declarator_type__

    @property
    def validator_type(self) -> type[EnumerationCallableDeclarator]:
        """The validator declarator type for this namespace, if any."""
        return self.__class__.__validator_type__

    @property
    def name(self) -> str:
        """The namespace display name."""
        return self.__class__.__namespace_name__

    def __call__(self, **kwargs: Any) -> Declarator:
        """Create a declarator instance for use in class assignments."""
        return self.declarator_type(**kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in {"declarator_type", "validator_type", "name"}:
            object.__setattr__(self, name, value)
            return
        self.bind(name, value)

    def _require_namespace(self) -> DeclarativeNamespace:
        dns = DECLARATIVE_NAMESPACE.get()
        if dns is None:
            raise RuntimeError(f"{self.name} namespace can only be used in declarative bodies.")
        return dns

    def declare(self, name: str, **kwargs: Any) -> Declarator:
        """Declare a named declarator without binding it as a class attribute."""
        dns = self._require_namespace()
        declarator = self.declarator_type(**kwargs)
        dns.register_declaration(declarator, name=name)
        return declarator

    def bind(self, name: str, value: Any) -> None:
        """Register a binding for a previously declared declarator."""
        dns = self._require_namespace()
        dns.register_binding(name, value, handle=self.name)

    def validator(self, *members: str, **kwargs: Any) -> Callable[[Callable], Declarator]:
        """Create a validation declarator as a decorator."""
        if self.validator_type is None:
            raise TypeError(f"{self.name} namespace does not support validators.")

        def decorator(fn: Callable) -> Declarator:
            return self.validator_type(callable=fn, members=members, **kwargs)

        return decorator


class MetadataNamespace(DeclaratorNamespace):
    """Namespace for metadata declarators and bindings."""
    __declarator_type__ = MetadataDeclarator
    __validator_type__ = MetadataValidator
    __namespace_name__ = "metadata"

    def __call__(
            self,
            default: Any = MISSING,
            factory: Callable[[], Any] | Missing = MISSING,
            validator: Callable[[Any], bool] | Missing = MISSING,
            doc: str | Missing = MISSING,
            config: Mapping[str, Any] | Missing = MISSING,
    ) -> MetadataDeclarator:
        """Create a declarator instance for use in class assignments."""
        return MetadataDeclarator(
            domain=True,
            default=default,
            factory=factory,
            validator=validator,
            doc=doc,
            config=config,
        )

    def declare(self, name: str, *, domain: bool = False, **kwargs: Any) -> Declarator:
        """Declare a named metadata declarator without binding it as a class attribute."""
        kwargs["domain"] = domain
        declarator = super().declare(name, **kwargs)
        return declarator


class OptionNamespace(DeclaratorNamespace):
    """Namespace for option declarators and bindings."""
    __declarator_type__ = OptionDeclarator
    __validator_type__ = OptionValidator
    __namespace_name__ = "option"


class NxFieldNamespace(DeclaratorNamespace):
    """Namespace for nxfield declarators and bindings."""
    __declarator_type__ = NxFieldDeclarator
    __validator_type__ = NxFieldValidator
    __namespace_name__ = "nxfield"

# endregion

# =============================================================================
# Singleton instances
# =============================================================================
# region Singleton instances


metadata = MetadataNamespace()
option = OptionNamespace()
nxfield = NxFieldNamespace()
meta = metadata

# endregion
