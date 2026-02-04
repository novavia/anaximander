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
from typing import ClassVar, Literal
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
    def report(cls, **kwargs) -> "Diagnostic":
        """Instantiate, emit, and collect a diagnostic in the active context.

        Args:
            **kwargs: Diagnostic initialization parameters.

        Returns:
            The reported diagnostic instance.

        Raises:
            AMLCompilationError: If the diagnostic is fatal.
        """
        # Instantiate the diagnostic before context selection or logging.
        diag = cls(**kwargs)
        # Select the current diagnostic bag with declarator-first precedence.
        bag = _select_bag()
        # Append to the bag if one is active.
        if bag is not None:
            bag.add(diag)
        # Emit a single structured log record.
        diag._emit()
        # Escalate immediately for fatal diagnostics.
        if diag.severity == "fatal":
            raise AMLCompilationError([diag])
        # Return the diagnostic for optional caller inspection.
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
    template = "Cannot redefine name '{name}' in declarative namespace."

    name: str


@dataclass(slots=True, kw_only=True)
class DeclarativeNamespaceDeletion(Diagnostic):
    """Report attempts to delete a name from a declarative namespace."""

    severity = "error"
    template = "Cannot delete items from a declarative namespace."


@dataclass(slots=True, kw_only=True)
class DeclaratorExpected(Diagnostic):
    """Report when a non-declarator is used where a declarator is required."""

    severity = "error"
    template = "Expected a Declarator instance, got {value!r}."

    value: object


@dataclass(slots=True, kw_only=True)
class DeclaratorNameInvalid(Diagnostic):
    """Report invalid declarator names."""

    severity = "error"
    template = "Declarator names cannot contain '.'."


@dataclass(slots=True, kw_only=True)
class DeclaratorNameConflict(Diagnostic):
    """Report duplicate declarator names."""

    severity = "error"
    template = "Duplicate declaration for name '{name}'."

    name: str


@dataclass(slots=True, kw_only=True)
class BindingAlreadyRegistered(Diagnostic):
    """Report duplicate bindings in a declarative namespace."""

    severity = "error"
    template = "Binding '{name}' is already registered in {scope}."

    name: str
    scope: str


@dataclass(slots=True, kw_only=True)
class UnnamedDeclaratorRegistration(Diagnostic):
    """Report declarators registered outside class assignment."""

    severity = "error"
    template = "Unnamed declarator registered outside class assignment."

# endregion

# =============================================================================
# Prototype diagnostics
# =============================================================================
# region Prototype diagnostics


@dataclass(slots=True, kw_only=True)
class InvalidPrototypeBase(Diagnostic):
    """Report invalid prototype base classes."""

    severity = "fatal"
    template = "Base class {base!r} is not a valid AML prototype base."

    base: object


@dataclass(slots=True, kw_only=True)
class MultipleInheritanceUnsupported(Diagnostic):
    """Report unsupported multiple inheritance for prototypes."""

    severity = "fatal"
    template = "Prototypes do not support multiple inheritance."


@dataclass(slots=True, kw_only=True)
class PrototypeInstantiationForbidden(Diagnostic):
    """Report attempts to instantiate AML prototypes directly."""

    severity = "fatal"
    template = "Archetypes, traits and prototypes cannot be instantiated directly."


@dataclass(slots=True, kw_only=True)
class ArchetypeTraitExpected(Diagnostic):
    """Report when a trait or archetype type is expected."""

    severity = "fatal"
    template = "Expected an {expected} type, got {actual!r}."

    expected: str
    actual: object


@dataclass(slots=True, kw_only=True)
class DeclaratorNotAllowedInArchetype(Diagnostic):
    """Report declarators that violate archetype constraints."""

    severity = "fatal"
    template = "{declarator!r} is not allowed in archetype {archetype!r}."

    declarator: object
    archetype: object


@dataclass(slots=True, kw_only=True)
class MetadataNotDeclared(Diagnostic):
    """Report metadata bindings for undeclared metadata."""

    severity = "fatal"
    template = "Metadata '{name}' is not declared for prototype {prototype!r}."

    name: str
    prototype: object


@dataclass(slots=True, kw_only=True)
class MetadataDomainBindingForbidden(Diagnostic):
    """Report domain metadata bindings in class headers."""

    severity = "fatal"
    template = "Cannot set domain metadata '{name}' in class header of {prototype!r}."

    name: str
    prototype: object


@dataclass(slots=True, kw_only=True)
class TraitSupertraitForbidden(Diagnostic):
    """Report invalid supertrait access."""

    severity = "fatal"
    template = "Only trait types have a supertrait."


@dataclass(slots=True, kw_only=True)
class InvalidViewSelection(Diagnostic):
    """Report invalid view selectors for trait/declarator/binding access."""

    severity = "fatal"
    template = "Invalid view '{view}'. Expected {expected}."

    view: str
    expected: str


@dataclass(slots=True, kw_only=True)
class AnnotatableDeclaratorUnnamed(Diagnostic):
    """Report annotatable declarators missing a name during annotation binding."""

    severity = "fatal"
    template = "Annotatable declarators must be named before annotation binding."


@dataclass(slots=True, kw_only=True)
class DeclaratorAnnotationMissing(Diagnostic):
    """Report missing annotations for annotatable declarators."""

    severity = "fatal"
    template = "Missing annotation for declarator {declarator!r}."

    declarator: object

# endregion

# =============================================================================
# Trait diagnostics
# =============================================================================
# region Trait diagnostics


@dataclass(slots=True, kw_only=True)
class TraitConformanceViolation(Diagnostic):
    """Report when a trait does not conform to an archetype."""

    severity = "fatal"
    template = "Trait {trait!r} does not conform to archetype {archetype!r}."

    trait: object
    archetype: object

# endregion
# =============================================================================
# Module diagnostics
# =============================================================================
# region Module diagnostics


@dataclass(slots=True, kw_only=True)
class ReferenceMissingOwnerMember(Diagnostic):
    """Report references that omit an owner/member component."""

    severity = "fatal"
    template = "Reference '{reference}' must include owner and member."

    reference: str


@dataclass(slots=True, kw_only=True)
class ReferenceKindInvalid(Diagnostic):
    """Report unknown reference kinds."""

    severity = "fatal"
    template = "Unknown reference kind '{kind}'."

    kind: str


@dataclass(slots=True, kw_only=True)
class ReferenceInvalid(Diagnostic):
    """Report invalid AML references."""

    severity = "fatal"
    template = "Invalid reference '{reference}'."

    reference: str


@dataclass(slots=True, kw_only=True)
class DisallowedModuleStatement(Diagnostic):
    """Report disallowed module or class-level statements."""

    severity = "fatal"
    template = "AML disallows statement: {statement}."

    statement: str


@dataclass(slots=True, kw_only=True)
class ModuleSourceMissing(Diagnostic):
    """Report modules without a resolvable source."""

    severity = "fatal"
    template = "Module {module!r} has no file origin to parse."

    module: object


@dataclass(slots=True, kw_only=True)
class TypeHintResolutionFailed(Diagnostic):
    """Report failures when resolving type hints."""

    severity = "fatal"
    template = "Failed to resolve type hints for {prototype!r}: {error}."

    prototype: object
    error: object


@dataclass(slots=True, kw_only=True)
class InvalidTypeRole(Diagnostic):
    """Report archetype/trait/prototype role mismatches."""

    severity = "fatal"
    template = "{prototype!r} must have role {role}."

    prototype: object
    role: str


@dataclass(slots=True, kw_only=True)
class PrototypeMetadescriptorForbidden(Diagnostic):
    """Report metadescriptor declarations on prototypes."""

    severity = "fatal"
    template = "Prototypes cannot declare metadescriptors."


@dataclass(slots=True, kw_only=True)
class BindingInvalidValue(Diagnostic):
    """Report invalid bound values for a declarator."""

    severity = "error"
    template = "Binding '{name}' with value '{value}' is not valid for {declarator!r} in {prototype!r}."  # noqa: E501

    name: str
    value: object
    declarator: object
    prototype: object


@dataclass(slots=True, kw_only=True)
class ValidatorFailed(Diagnostic):
    """Report failed validator checks."""

    severity = "error"
    template = "Binding '{name}' with value '{value}' failed validation by {validator!r} in {prototype!r}."  # noqa: E501

    name: str
    value: object
    validator: object
    prototype: object


@dataclass(slots=True, kw_only=True)
class PrototypeValidatorFailed(Diagnostic):
    """Report failed prototype validators."""

    severity = "error"
    template = "Prototype {prototype!r} failed validation by {validator!r}."

    prototype: object
    validator: object


@dataclass(slots=True, kw_only=True)
class EnumerationInvalid(Diagnostic):
    """Report invalid enumeration members or configuration."""

    severity = "error"
    template = "{message}"

    message: str

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
def diagnostic_context(bag: DiagnosticBag, selector: ContextVar[DiagnosticBag | None]):
    """Temporarily bind a diagnostic bag to a context variable."""
    token = selector.set(bag)
    try:
        yield
    finally:
        selector.reset(token)

# endregion
