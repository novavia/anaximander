# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Define the AML diagnostic model, context selection, and escalation hooks.

This module provides the foundational building blocks for structured diagnostics
in AML. Diagnostics are lightweight data objects with stable severity, optional
message templates, and a standard reporting mechanism. Diagnostic bags collect
issues for semantic owners, while context variables select the active bag at
report time. Fatal diagnostics raise immediately; phase boundaries are expected
to inspect bags and raise aggregated errors when needed.
"""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from __future__ import annotations

import weakref
from contextvars import ContextVar
from dataclasses import dataclass, fields
from functools import wraps
from typing import Any, Callable, ClassVar, Literal, TypeVar, cast
from contextlib import contextmanager

from loguru import logger

# endregion

# =============================================================================
# Constants
# =============================================================================
# region Constants

Severity = Literal["warning", "error", "fatal"]

LOGURU_LEVEL: dict[Severity, str] = {
    "warning": "WARNING",
    "error": "ERROR",
    "fatal": "CRITICAL",
}

# endregion

# =============================================================================
# Context variables
# =============================================================================
# region Context variables

PROJECT_DIAGNOSTICS: ContextVar["DiagnosticBag | None"] = ContextVar(
    "PROJECT_DIAGNOSTICS",
    default=None,
)
MODULE_DIAGNOSTICS: ContextVar["DiagnosticBag | None"] = ContextVar(
    "MODULE_DIAGNOSTICS",
    default=None,
)
PROTOTYPE_DIAGNOSTICS: ContextVar["DiagnosticBag | None"] = ContextVar(
    "PROTOTYPE_DIAGNOSTICS",
    default=None,
)
DECLARATOR_DIAGNOSTICS: ContextVar["DiagnosticBag | None"] = ContextVar(
    "DECLARATOR_DIAGNOSTICS",
    default=None,
)

# endregion

# =============================================================================
# Exceptions
# =============================================================================
# region Exceptions


class AMLCompilationError(RuntimeError):
    """Aggregate AML compilation diagnostics into a single exception."""

    def __init__(self, diagnostics: list["Diagnostic"], message: str | None = None):
        """Initialize the exception with a diagnostic payload.

        Args:
            diagnostics: Diagnostics that triggered the compilation failure.
            message: Optional override message for the exception.
        """
        super().__init__(message or f"AML compilation failed with {len(diagnostics)} issue(s).")
        self.diagnostics = diagnostics

# endregion

# =============================================================================
# Diagnostic model
# =============================================================================
# region Diagnostic model


@dataclass(slots=True)
class Diagnostic:
    """Represent a structured AML diagnostic message."""

    severity: ClassVar[Severity]
    template: ClassVar[str | None] = None

    _message: str | None = None

    @property
    def message(self) -> str:
        """Return the rendered diagnostic message."""
        if self._message is not None:
            return self._message
        if self.template is not None:
            # Merge instance attributes with dataclass fields for template payloads.
            payload: dict[str, object] = {}
            try:
                payload.update(vars(self))
            except TypeError:
                # slots-only instances may not have __dict__.
                pass
            for field in fields(self):
                payload.setdefault(field.name, getattr(self, field.name))
            return self.template.format(**payload)
        raise ValueError("Diagnostic has neither message nor template")

    @classmethod
    def report(cls, message: str | None = None, **kwargs) -> "Diagnostic":
        """Instantiate, emit, and collect a diagnostic in the active context.

        Args:
            message: Explicit diagnostic message override.
            **kwargs: Diagnostic initialization parameters.

        Returns:
            The reported diagnostic instance.

        Raises:
            AMLCompilationError: If the diagnostic is fatal.
        """
        # Instantiate the diagnostic before context selection or logging.
        if message is not None:
            kwargs.setdefault("_message", message)
        if "message" in kwargs:
            kwargs.setdefault("_message", kwargs["message"])
            kwargs.pop("message", None)
        if "_message" in kwargs:
            try:
                diag = cls(**kwargs)
            except TypeError:
                diag = cls(kwargs["_message"])  # type: ignore[call-arg]
            return cls._finalize_report(diag)
        diag = cls(**kwargs)
        return cls._finalize_report(diag)

    @classmethod
    def _finalize_report(cls, diag: "Diagnostic") -> "Diagnostic":
        """Attach, emit, and escalate a diagnostic after instantiation."""
        bag = _select_bag()
        if bag is not None:
            bag.add(diag)
        diag._emit()
        if diag.severity == "fatal":
            raise AMLCompilationError([diag])
        return diag

    def _emit(self) -> None:
        """Emit a single structured log record for the diagnostic."""
        logger.bind(
            severity=self.severity,
            diagnostic=type(self).__name__,
            context=current_context_path(),
        ).log(
            LOGURU_LEVEL[self.severity],
            self.message,
        )

# endregion

# =============================================================================
# Declarative namespace diagnostics
# =============================================================================
# region Declarative namespace diagnostics


@dataclass(slots=True, kw_only=True)
class DeclarativeNamespaceRedefinition(Diagnostic):
    """Report attempts to redefine a name in a declarative namespace."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class DeclarativeNamespaceDeletion(Diagnostic):
    """Report attempts to delete a name from a declarative namespace."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorExpected(Diagnostic):
    """Report when a non-declarator is used where a declarator is required."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class DeclaratorNameInvalid(Diagnostic):
    """Report invalid declarator names."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorNameConflict(Diagnostic):
    """Report duplicate declarator names."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class BindingAlreadyRegistered(Diagnostic):
    """Report duplicate bindings in a declarative namespace."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class UnnamedDeclaratorRegistration(Diagnostic):
    """Report declarators registered outside class assignment."""

    severity = "error"

# endregion

# =============================================================================
# Declarator diagnostics
# =============================================================================
# region Declarator diagnostics


@dataclass(slots=True, kw_only=True)
class DeclaratorInvariantViolation(Diagnostic):
    """Report internal declarator invariant violations."""

    severity = "fatal"


@dataclass(slots=True, kw_only=True)
class DeclaratorHandleInvalid(Diagnostic):
    """Report invalid declarator handles."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorHandleConflict(Diagnostic):
    """Report conflicting declarator handles."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorTypeConstraintViolation(Diagnostic):
    """Report declarator type/shape constraints."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorNullabilityViolation(Diagnostic):
    """Report nullability violations for declarators."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorOverrideViolation(Diagnostic):
    """Report invalid declarator overrides or bindings."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorReassignmentViolation(Diagnostic):
    """Report declarator reassignment violations."""

    severity = "error"

# endregion

# =============================================================================
# Prototype diagnostics
# =============================================================================
# region Prototype diagnostics


@dataclass(slots=True, kw_only=True)
class InvalidPrototypeBase(Diagnostic):
    """Report invalid prototype base classes."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class MultipleInheritanceUnsupported(Diagnostic):
    """Report unsupported multiple inheritance for prototypes."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class PrototypeInstantiationForbidden(Diagnostic):
    """Report attempts to instantiate AML prototypes directly."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class ArchetypeTraitExpected(Diagnostic):
    """Report when a trait or archetype type is expected."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class DeclaratorNotAllowedInArchetype(Diagnostic):
    """Report declarators that violate archetype constraints."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class MetadataNotDeclared(Diagnostic):
    """Report metadata bindings for undeclared metadata."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class MetadataDomainBindingForbidden(Diagnostic):
    """Report domain metadata bindings in class headers."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class TraitSupertraitForbidden(Diagnostic):
    """Report invalid supertrait access."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class InvalidViewSelection(Diagnostic):
    """Report invalid view selectors for trait/declarator/binding access."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class AnnotatableDeclaratorUnnamed(Diagnostic):
    """Report annotatable declarators missing a name during annotation binding."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class DeclaratorAnnotationMissing(Diagnostic):
    """Report missing annotations for annotatable declarators."""

    severity = "error"


# endregion

# =============================================================================
# Trait diagnostics
# =============================================================================
# region Trait diagnostics


@dataclass(slots=True, kw_only=True)
class TraitConformanceViolation(Diagnostic):
    """Report when a trait does not conform to an archetype."""

    severity = "error"


# endregion
# =============================================================================
# Module diagnostics
# =============================================================================
# region Module diagnostics


@dataclass(slots=True, kw_only=True)
class ReferenceMissingOwnerMember(Diagnostic):
    """Report references that omit an owner/member component."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class ReferenceKindInvalid(Diagnostic):
    """Report unknown reference kinds."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class ReferenceInvalid(Diagnostic):
    """Report invalid AML references."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class DisallowedModuleStatement(Diagnostic):
    """Report disallowed module or class-level statements."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class ModuleSourceMissing(Diagnostic):
    """Report modules without a resolvable source."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class TypeHintResolutionFailed(Diagnostic):
    """Report failures when resolving type hints."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class InvalidTypeRole(Diagnostic):
    """Report archetype/trait/prototype role mismatches."""

    severity = "error"



@dataclass(slots=True, kw_only=True)
class PrototypeMetadescriptorForbidden(Diagnostic):
    """Report metadescriptor declarations on prototypes."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class BindingInvalidValue(Diagnostic):
    """Report invalid bound values for a declarator."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class ValidatorFailed(Diagnostic):
    """Report failed validator checks."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class PrototypeValidatorFailed(Diagnostic):
    """Report failed prototype validators."""

    severity = "error"


@dataclass(slots=True, kw_only=True)
class EnumerationInvalid(Diagnostic):
    """Report invalid enumeration members or configuration."""

    severity = "error"

# endregion

# =============================================================================
# Diagnostic bag
# =============================================================================
# region Diagnostic bag


class DiagnosticBag:
    """Collect diagnostics associated with a semantic owner."""

    def __init__(self, owner: object):
        """Initialize the bag with a weak owner reference.

        Args:
            owner: The semantic owner associated with the bag.
        """
        # Keep a weak reference to avoid reference cycles.
        self._owner_ref = weakref.ref(owner)
        # Store collected diagnostics in insertion order.
        self._messages: list[Diagnostic] = []

    @property
    def owner(self) -> object | None:
        """Return the owning object, if it is still alive."""
        return self._owner_ref()

    def rebind_owner(self, owner: object) -> None:
        """Rebind the bag to a new owner."""
        self._owner_ref = weakref.ref(owner)

    def add(self, diag: Diagnostic) -> None:
        """Append a diagnostic to the bag."""
        # Append diagnostics in the order they are reported.
        self._messages.append(diag)

    @property
    def messages(self) -> list[Diagnostic]:
        """Return collected diagnostics."""
        return self._messages

# endregion

# =============================================================================
# Diagnostic bag specializations
# =============================================================================
# region Diagnostic bag specializations


class ProjectDiagnosticBag(DiagnosticBag):
    """Diagnostic bag for project scope."""

    def __repr__(self) -> str:
        owner = self.owner
        name = getattr(owner, "name", None) if owner is not None else None
        return f"<ProjectDiagnosticBag {name or 'unbound'}>"


class ModuleDiagnosticBag(DiagnosticBag):
    """Diagnostic bag for module scope."""

    def __repr__(self) -> str:
        owner = self.owner
        name = getattr(owner, "__name__", None) if owner is not None else None
        return f"<ModuleDiagnosticBag {name or 'unbound'}>"


class PrototypeDiagnosticBag(DiagnosticBag):
    """Diagnostic bag for prototype scope with provisional labeling support."""

    def __init__(self, owner: object, *, provisional_name: str | None = None):
        super().__init__(owner)
        self._provisional_name = provisional_name

    def __repr__(self) -> str:
        owner = self.owner
        if owner is None:
            name = self._provisional_name or "unbound"
        else:
            name = getattr(owner, "__name__", None) or self._provisional_name or "unbound"
        return f"<PrototypeDiagnosticBag {name}>"


class DeclaratorDiagnosticBag(DiagnosticBag):
    """Diagnostic bag for declarator scope."""

    def __repr__(self) -> str:
        owner = self.owner
        if owner is None:
            return "<DeclaratorDiagnosticBag unbound>"
        name = getattr(owner, "name", None)
        owner_name = getattr(getattr(owner, "owner", None), "__name__", None)
        if name and owner_name:
            return f"<DeclaratorDiagnosticBag {owner_name}.{name}>"
        if name:
            return f"<DeclaratorDiagnosticBag {name}>"
        return "<DeclaratorDiagnosticBag>"

# endregion

# =============================================================================
# Context helpers
# =============================================================================
# region Context helpers


def current_context_path() -> str | None:
    """Return a serialized context path for the active diagnostic bag."""
    # Resolve the active diagnostic bag for context metadata.
    bag = _select_bag()
    if bag is None:
        return None
    # Extract the owner for best-effort naming.
    owner = bag.owner
    if owner is None:
        return None
    # Prefer module-qualified names for type owners.
    owner_name = getattr(owner, "__name__", None)
    owner_module = getattr(owner, "__module__", None)
    if owner_name and owner_module:
        return f"{owner_module}.{owner_name}"
    if owner_name:
        return owner_name
    return repr(owner)

# endregion

# =============================================================================
# Context selection
# =============================================================================
# region Context selection


def _select_bag() -> "DiagnosticBag | None":
    """Return the active diagnostic bag based on precedence rules."""
    # Select the most specific active bag, falling back to project scope.
    return (
        DECLARATOR_DIAGNOSTICS.get()
        or PROTOTYPE_DIAGNOSTICS.get()
        or MODULE_DIAGNOSTICS.get()
        or PROJECT_DIAGNOSTICS.get()
    )

# endregion

# =============================================================================
# Escalation helpers
# =============================================================================
# region Escalation helpers


def raise_on_errors(bag: DiagnosticBag) -> None:
    """Raise an AMLCompilationError if a bag contains error or fatal diagnostics."""
    # Gather non-fatal error diagnostics for phase-boundary escalation.
    errors = [diag for diag in bag.messages if diag.severity in {"error", "fatal"}]
    if errors:
        raise AMLCompilationError(errors)

# endregion

# =============================================================================
# Context managers
# =============================================================================
# region Context managers


@contextmanager
def diagnostic_context(owner: object):
    """Temporarily bind diagnostics based on the owner type."""
    bag, selector = _resolve_context(owner)
    token = selector.set(bag)
    try:
        yield
    finally:
        selector.reset(token)

# endregion

# =============================================================================
# Context helpers
# =============================================================================
# region Context helpers


def _resolve_context(owner: object) -> tuple[DiagnosticBag, ContextVar[DiagnosticBag | None]]:
    """Resolve the diagnostic bag and selector for a given owner."""
    if hasattr(owner, "__diagnostics__"):
        bag = owner.__diagnostics__
    elif isinstance(owner, DiagnosticBag):
        bag = owner
    else:
        raise TypeError(f"Unsupported diagnostics owner: {owner!r}.")
    if isinstance(bag, DeclaratorDiagnosticBag):
        return bag, DECLARATOR_DIAGNOSTICS
    if isinstance(bag, PrototypeDiagnosticBag):
        return bag, PROTOTYPE_DIAGNOSTICS
    if isinstance(bag, ModuleDiagnosticBag):
        return bag, MODULE_DIAGNOSTICS
    if isinstance(bag, ProjectDiagnosticBag):
        return bag, PROJECT_DIAGNOSTICS
    raise TypeError(f"Unsupported diagnostics bag: {bag!r}.")


F = TypeVar("F", bound=Callable[..., Any])


def with_diagnostics(fn: F) -> F:
    """Decorate a method to apply diagnostic context for its owner."""

    @wraps(fn)
    def wrapper(owner, *args, **kwargs):
        with diagnostic_context(owner):
            return fn(owner, *args, **kwargs)

    return cast(F, wrapper)

# endregion
