"""This module defines the Protodescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import re
from typing import cast

from .declarative import (
    AssignableDeclarator,
    CallableDeclarator,
    Declarator,
    DeclaratorRegistry,
    MultiRegistry,
    declarator,
)

# endregion

# =============================================================================
# Metadescriptor classes
# =============================================================================
# region Metadescriptor classes


@declarator
class Metadescriptor(Declarator):
    """Base class for prototype-level descriptors declared in archetypes and traits."""
    __handle__ = "meta"
    __reserved_patterns__ = {re.compile(r"^nx.*")}


@declarator
class MetadataDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for metadata fields."""
    __handle__ = "metadata"


@declarator
class OptionDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for option fields."""
    __handle__ = "option"


@declarator
class NxFieldDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"


@declarator
class MetadataValidator(CallableDeclarator, Metadescriptor):
    """The metadescriptor class for metadata validators."""
    __handle__ = "metavalidator"

# endregion

# =============================================================================
# Registry classes
# =============================================================================
# region Registry classes


class MetadescriptorRegistry(MultiRegistry):
    """Registry for metadescriptors."""

    __namespaces__ = {"metadata", "nxfield", "option"}

    def __init__(self):
        metadata = DeclaratorRegistry[MetadataDeclarator]()
        nxfield = DeclaratorRegistry[NxFieldDeclarator]()
        option = DeclaratorRegistry[OptionDeclarator]()
        super().__init__(metadata=metadata, nxfield=nxfield, option=option)

    @property
    def metadata(self) -> DeclaratorRegistry[MetadataDeclarator]:
        """Returns the metadata metadescriptor registry."""
        return cast(DeclaratorRegistry[MetadataDeclarator], self._data["metadata"])

    @property
    def nxfield(self) -> DeclaratorRegistry[NxFieldDeclarator]:
        """Returns the nxfield metadescriptor registry."""
        return cast(DeclaratorRegistry[NxFieldDeclarator], self._data["nxfield"])

    @property
    def option(self) -> DeclaratorRegistry[OptionDeclarator]:
        """Returns the option metadescriptor registry."""
        return cast(DeclaratorRegistry[OptionDeclarator], self._data["option"])

    def register(self, key, item: Metadescriptor, *, namespace: str | None = None) -> None:
        """Registers a metadescriptor in the appropriate registry."""
        if namespace is not None:
            registry = self.get_registry(namespace)
            registry.register(key, item)
            return
        match item:
            case MetadataDeclarator():
                self.metadata.register(key, item)
            case NxFieldDeclarator():
                self.nxfield.register(key, item)
            case OptionDeclarator():
                self.option.register(key, item)
            case _:
                msg = f"Cannot register metadescriptor of type {type(item).__name__}."
                raise TypeError(msg)

# endregion
