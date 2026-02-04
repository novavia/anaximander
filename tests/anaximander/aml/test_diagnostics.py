# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Tests for AML diagnostics scaffolding."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import pytest

from anaximander.aml.diagnostics import (
    AMLCompilationError,
    DECLARATOR_DIAGNOSTICS,
    DeclaratorDiagnosticBag,
    MODULE_DIAGNOSTICS,
    PROJECT_DIAGNOSTICS,
    PROTOTYPE_DIAGNOSTICS,
    Diagnostic,
    DiagnosticBag,
    ModuleDiagnosticBag,
    ProjectDiagnosticBag,
    PrototypeDiagnosticBag,
    diagnostic_context,
)

# endregion

# =============================================================================
# Helpers
# =============================================================================
# region Helpers


class DummyOwner:
    """Simple owner class for diagnostic bag tests."""


class WarningDiagnostic(Diagnostic):
    """Diagnostic with warning severity and a template message."""

    severity = "warning"
    template = "Name '{name}' is questionable."

    def __init__(self, name: str, message: str | None = None):
        """Initialize the warning diagnostic with payload values."""
        self.name = name
        self._message = message


class FatalDiagnostic(Diagnostic):
    """Diagnostic with fatal severity and explicit message override."""

    severity = "fatal"

    def __init__(self, message: str):
        """Initialize the fatal diagnostic with explicit message."""
        self._message = message


# =============================================================================
# Fixtures
# =============================================================================
# region Fixtures


@pytest.fixture()
def warning_bag() -> DiagnosticBag:
    """Provide a diagnostic bag bound to a dummy owner."""
    return DeclaratorDiagnosticBag(DummyOwner())

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_report_collects_and_formats_message(warning_bag):
    """Report a warning diagnostic and collect it in the active bag."""
    token = PROJECT_DIAGNOSTICS.set(warning_bag)
    try:
        diag = WarningDiagnostic.report(name="alpha")
    finally:
        PROJECT_DIAGNOSTICS.reset(token)
    assert diag.message == "Name 'alpha' is questionable."
    assert warning_bag.messages == [diag]


def test_fatal_diagnostic_escalates(warning_bag):
    """Fatal diagnostics raise immediately while still reporting."""
    token = PROJECT_DIAGNOSTICS.set(warning_bag)
    try:
        with pytest.raises(AMLCompilationError) as excinfo:
            FatalDiagnostic.report(message="Stop now.")
    finally:
        PROJECT_DIAGNOSTICS.reset(token)
    assert len(excinfo.value.diagnostics) == 1
    assert excinfo.value.diagnostics[0].message == "Stop now."


def test_context_precedence_selects_declarator(warning_bag):
    """Declarator diagnostics should override broader context bags."""
    proto_bag = PrototypeDiagnosticBag(DummyOwner())
    module_bag = ModuleDiagnosticBag(DummyOwner())
    project_bag = ProjectDiagnosticBag(DummyOwner())
    token_project = PROJECT_DIAGNOSTICS.set(project_bag)
    token_module = MODULE_DIAGNOSTICS.set(module_bag)
    token_proto = PROTOTYPE_DIAGNOSTICS.set(proto_bag)
    token_decl = DECLARATOR_DIAGNOSTICS.set(warning_bag)
    try:
        diag = WarningDiagnostic.report(name="beta")
    finally:
        DECLARATOR_DIAGNOSTICS.reset(token_decl)
        PROTOTYPE_DIAGNOSTICS.reset(token_proto)
        MODULE_DIAGNOSTICS.reset(token_module)
        PROJECT_DIAGNOSTICS.reset(token_project)
    assert warning_bag.messages == [diag]
    assert proto_bag.messages == []
    assert module_bag.messages == []
    assert project_bag.messages == []


def test_diagnostic_context_overrides_project(warning_bag):
    """Diagnostic context should override broader project scope."""
    project_bag = ProjectDiagnosticBag(DummyOwner())
    token_project = PROJECT_DIAGNOSTICS.set(project_bag)
    try:
        with diagnostic_context(warning_bag):
            diag = WarningDiagnostic.report(name="gamma")
    finally:
        PROJECT_DIAGNOSTICS.reset(token_project)
    assert warning_bag.messages == [diag]
    assert project_bag.messages == []


def test_diagnostic_context_nested_precedence(warning_bag):
    """Nested diagnostic contexts should select the most specific bag."""
    proto_bag = PrototypeDiagnosticBag(DummyOwner())
    module_bag = ModuleDiagnosticBag(DummyOwner())
    project_bag = ProjectDiagnosticBag(DummyOwner())
    token_project = PROJECT_DIAGNOSTICS.set(project_bag)
    try:
        with diagnostic_context(module_bag):
            with diagnostic_context(proto_bag):
                with diagnostic_context(warning_bag):
                    diag = WarningDiagnostic.report(name="delta")
    finally:
        PROJECT_DIAGNOSTICS.reset(token_project)
    assert warning_bag.messages == [diag]
    assert proto_bag.messages == []
    assert module_bag.messages == []
    assert project_bag.messages == []

# endregion
