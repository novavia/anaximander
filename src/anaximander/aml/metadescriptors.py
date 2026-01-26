"""This module defines the Metadescriptor classes for the Anaximander Modeling Language (AML)."""

# =============================================================================
# Imports
# =============================================================================
# region Imports


import re
from collections.abc import Callable
from typing import Any, cast

from attrs import field

from .declarative import (
    MISSING,
    AssignableDeclarator,
    CallableDeclarator,
    Declarator,
    DeclaratorRegistry,
    EnumerationCallableDeclarator,
    MultiRegistry,
    _tighten_bound_value,
    _tighten_classvar,
    _tighten_nullable,
    _tighten_type,
    declarative,
    declarator,
    is_not_missing,
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
class AssignableMetadescriptor(AssignableDeclarator, Metadescriptor):
    """Base class for assignable prototype-level descriptors declared in archetypes and traits."""
    pass


@declarator
class MetadataDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for metadata fields."""
    __handle__ = "metadata"
    domain: bool = field(default=False)  # Whether this metadata is part of the domain schema  # noqa

    @property
    def domain_bindable(self) -> bool:
        """Whether this declarator instance supports binding to values."""
        return self.domain

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("domain", self.domain, bool)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind a metadata value, enforcing monotone tightening and inline validation."""
        if is_not_missing(previous) and value != previous:
            if not _tighten_bound_value(value, previous):
                raise AttributeError(
                    f"Metadata '{self.name}' cannot be reassigned with a looser value."
                )
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            raise ValueError(
                f"Inline validation failed for metadata '{self.name}' with value: {value}"
            )
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten type constraints only."""
        if type(override) is not type(self):
            raise AttributeError("Metadata declarators can only be overridden by the same class.")
        if override == self:
            return
        if self.domain != override.domain:
            raise AttributeError("Metadata declarator 'domain' cannot be overridden.")
        if not _tighten_type(self.type, override.type):
            raise AttributeError("Metadata declarator type cannot be loosened.")
        if not _tighten_nullable(self.nullable, override.nullable):
            raise AttributeError("Metadata declarator nullability cannot be loosened.")
        if not _tighten_classvar(self.classvar, override.classvar):
            raise AttributeError("Metadata declarator classvar cannot be overridden.")

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate a metadata binding in the context of a host type."""
        if value is None and self.nullable is False:
            return False
        if self.type is not None and value is not None:
            if not isinstance(value, self.type):
                if not (isinstance(value, type) and issubclass(value, self.type)):
                    return False
        return True


@declarator
class OptionDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for option fields."""
    __handle__ = "option"


@declarator
class NxFieldDeclarator(AssignableMetadescriptor):
    """The metadescriptor class for nxfield, i.e. abstract semantic fields."""
    __handle__ = "nxfield"
    fieldtype: type = field()

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("fieldtype", self.fieldtype, type)

    def __bind__(self, value: Any, previous: Any = MISSING) -> None:
        """Bind an nxfield value, disallowing reassignment in derived prototypes."""
        if is_not_missing(previous) and value != previous:
            raise AttributeError(f"NxField '{self.name}' cannot be reassigned.")
        validator = self.validator
        if is_not_missing(validator) and not validator(value):
            raise ValueError(
                f"Inline validation failed for nxfield '{self.name}' with value: {value}"
            )
        return value

    def __override__(self, override: Declarator) -> None:
        """Allow overrides that tighten fieldtype/type constraints only."""
        if type(override) is not type(self):
            raise AttributeError("NxField declarators can only be overridden by the same class.")
        if override == self:
            return
        if not _tighten_type(self.fieldtype, override.fieldtype):
            raise AttributeError("NxField declarator fieldtype cannot be loosened.")
        if not _tighten_type(self.type, override.type):
            raise AttributeError("NxField declarator type cannot be loosened.")
        if not _tighten_nullable(self.nullable, override.nullable):
            raise AttributeError("NxField declarator nullability cannot be loosened.")
        if not _tighten_classvar(self.classvar, override.classvar):
            raise AttributeError("NxField declarator classvar cannot be overridden.")

    def __validate_binding__(self, host: type, value: Any) -> bool:
        """Validate an nxfield binding in the context of a host type."""
        return True


@declarator
class PrototypeValidator(CallableDeclarator[Callable[[declarative], bool]], Metadescriptor):
    """The metadescriptor class for prototype validators."""
    __handle__ = "prototype_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class MetadataValidator(EnumerationCallableDeclarator[Callable[[declarative, Any], bool]], Metadescriptor):  # noqa
    """The metadescriptor class for metadata validators."""
    __handle__ = "metadata_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class OptionValidator(EnumerationCallableDeclarator[Callable[[declarative, Any], bool]], Metadescriptor):  # noqa
    """The metadescriptor class for option validators."""
    __handle__ = "option_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


@declarator
class NxFieldValidator(EnumerationCallableDeclarator[Callable[[declarative, Any], bool]], Metadescriptor):  # noqa
    """The metadescriptor class for nxfield validators."""
    __handle__ = "nxfield_validator"

    def __validate__(self) -> None:
        super().__validate__()
        self._validate_value_type("callable", self.callable, Callable)  # type: ignore[arg-type]


# endregion

# =============================================================================
# Registry classes
# =============================================================================
# region Registry classes


class MetadescriptorRegistry(MultiRegistry):
    """Registry for metadescriptors."""

    __handles__ = {"metadata", "nxfield", "option", "validation"}

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

    def register(self, key, item: Metadescriptor, *, handle: str | None = None) -> None:
        """Registers a metadescriptor in the appropriate registry."""
        if handle is not None:
            registry = self.get_registry(handle)
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
