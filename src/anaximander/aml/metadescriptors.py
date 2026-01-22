"""This module defines the Metadescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import re
from typing import Any, Callable, cast

from attrs import field

from .declarative import (
    AssignableDeclarator,
    CallableDeclarator,
    Declarator,
    DeclaratorRegistry,
    EnumerationCallableDeclarator,
    MultiRegistry,
    declarative,
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
    domain: bool = field(default=None)  # Whether this metadata is part of the domain schema  # noqa


@declarator
class OptionDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for option fields."""
    __handle__ = "option"


@declarator
class NxFieldDeclarator(AssignableDeclarator, Metadescriptor):
    """The metadescriptor class for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"
    fieldtype: type = field()


@declarator
class PrototypeValidator(CallableDeclarator[Callable[[declarative], bool]], Metadescriptor):
    """The metadescriptor class for prototype validators."""
    __handle__ = "prototype_validator"


@declarator
class MetadataValidator(EnumerationCallableDeclarator[Callable[[declarative, Any], bool]], Metadescriptor):  # noqa
    """The metadescriptor class for metadata validators."""
    __handle__ = "metadata_validator"
    callable: Callable[[declarative, Any], bool] = field()


@declarator
class OptionValidator(EnumerationCallableDeclarator[Callable[[declarative, Any], bool]], Metadescriptor):  # noqa
    """The metadescriptor class for option validators."""
    __handle__ = "option_validator"
    callable: Callable[[declarative, Any], bool] = field()


@declarator
class NxFieldValidator(EnumerationCallableDeclarator[Callable[[declarative, Any], bool]], Metadescriptor):  # noqa
    """The metadescriptor class for nxfield validators."""
    __handle__ = "nxfield_validator"
    callable: Callable[[declarative, Any], bool] = field()


# endregion

# =============================================================================
# Registry classes
# =============================================================================
# region Registry classes


class MetadescriptorRegistry(MultiRegistry):
    """Registry for metadescriptors."""

    __namespaces__ = {"metadata", "nxfield", "option", "validation"}

    def __init__(self):
        metadata = DeclaratorRegistry[MetadataDeclarator]()
        nxfield = DeclaratorRegistry[NxFieldDeclarator]()
        option = DeclaratorRegistry[OptionDeclarator]()
        validation = DeclaratorRegistry[CallableDeclarator]()
        super().__init__(metadata=metadata, nxfield=nxfield, option=option, validation=validation)

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

    @property
    def validation(self) -> DeclaratorRegistry[CallableDeclarator]:
        """Returns the validation metadescriptor registry."""
        return cast(DeclaratorRegistry[CallableDeclarator], self._data["validation"])

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
            case PrototypeValidator() | MetadataValidator() | OptionValidator() | NxFieldValidator():  # noqa
                self.validation.register(key, item)

# endregion
