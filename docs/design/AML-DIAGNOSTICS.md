# AML Diagnostics and Context Specification

## Scope

This document specifies how diagnostics are represented, emitted, contextualized, collected, rendered, and escalated in the Anaximander Modeling Language (AML).

The goals are:

- Structured, typed diagnostics (not ad-hoc exceptions)
- Deterministic association of diagnostics to semantic owners
- Multi-diagnostic reporting per compilation phase
- Clear failure semantics at phase boundaries
- Minimal infrastructure and good developer experience

This specification is normative for all AML compiler, validation, and transformation code.

## Diagnostic Model

A diagnostic represents a semantic issue detected during AML processing.

Diagnostics are data objects with light behavior. They are responsible for:

- carrying structured information
- producing a human-readable message
- reporting themselves into the active diagnostic context

Diagnostics do not encode control-flow policy beyond the distinction between non-fatal and fatal.

Severity is determined by the diagnostic class, not by runtime configuration.

Allowed severities:

- warning: AML is valid but questionable
- error: AML is invalid, but processing may continue
- fatal: AML invariant violation; processing must abort

Severity must not be inferred from log filtering.

## Diagnostic Base Class

Diagnostics use a lightweight data model with built-in defaults.
The reference implementation uses dataclasses, though attrs is also acceptable.
Pydantic is explicitly excluded at this level to avoid runtime and dependency coupling.

The Diagnostic base class supports two mutually compatible ways of specifying a message:

- a class-level template formatted with keyword arguments
- an explicit instance-level message override

The rendered message is always accessed via the message property.

```python
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional
from loguru import logger

Severity = Literal["warning", "error", "fatal"]

LOGURU_LEVEL = {
    "warning": "WARNING",
    "error": "ERROR",
    "fatal": "CRITICAL",
}

@dataclass(slots=True)
class Diagnostic:
    severity: ClassVar[Severity]
    template: ClassVar[Optional[str]] = None

    _message: Optional[str] = None

    @property
    def message(self) -> str:
        if self._message is not None:
            return self._message
        if self.template is not None:
            return self.template.format(**self.__dict__)
        raise ValueError("Diagnostic has neither message nor template")

    @classmethod
    def report(cls, **kwargs) -> "Diagnostic":
        diag = cls(**kwargs)

        bag = (
            DECLARATOR_DIAGNOSTICS.get()
            or PROTOTYPE_DIAGNOSTICS.get()
            or MODULE_DIAGNOSTICS.get()
            or PROJECT_DIAGNOSTICS.get()
        )

        if bag is not None:
            bag.add(diag)

        diag._emit()

        if diag.severity == "fatal":
            raise AMLCompilationError([diag])

        return diag

    def _emit(self) -> None:
        logger.bind(
            severity=self.severity,
            diagnostic=type(self).__name__,
            context=current_context_path(),
        ).log(
            LOGURU_LEVEL[self.severity],
            self.message,
        )
```

## Diagnostic Bags

Each DiagnosticBag owns collected diagnostics but holds only a weak reference
to its semantic owner to avoid reference cycles.

```python
import weakref

class DiagnosticBag:
    def __init__(self, owner):
        self._owner_ref = weakref.ref(owner)
        self._messages: list[Diagnostic] = []

    @property
    def owner(self):
        return self._owner_ref()

    def add(self, diag: Diagnostic):
        self._messages.append(diag)

    @property
    def messages(self) -> list[Diagnostic]:
        return self._messages
```

## Context Variables

Diagnostics are contextualized using ContextVars that select the active DiagnosticBag.

```
PROJECT_DIAGNOSTICS
MODULE_DIAGNOSTICS
PROTOTYPE_DIAGNOSTICS
DECLARATOR_DIAGNOSTICS
```

ContextVars act only as selectors and do not manage bag lifetimes.

## Logging Contract

For each reported diagnostic, exactly one log record is emitted.

Structured fields:

- severity
- diagnostic
- context

Log record message:

- rendered diagnostic message

The logging backend must not affect semantic correctness.

## Escalation Rules

- Fatal diagnostics raise immediately
- Warning and error diagnostics never raise at report time
- Phase boundaries are responsible for inspecting bags and raising if needed

## Non-Goals

This specification does not define error codes, localization, IDE integration,
or advanced policy engines.
