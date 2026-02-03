# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise forward-reference handling in AML module finalization."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from tests.anaximander.aml.modules import type_hints_module as thm

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_finalize_module_resolves_forward_refs(aml_finalize):
    """Resolve forward references in module type hints."""
    aml_finalize(thm)
    field = thm.Sensor.declarators("merged").field["temperature"]
    assert field.type is thm.Temperature
    assert field.nullable is True

# endregion
