# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Smoke tests for AML module finalization."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from anaximander.aml.modules import NxModuleType
from tests.anaximander.aml.modules import value_parsers_module as vpm

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_smoke_finalize_module(aml_finalize):
    """Finalize a sample AML module and inspect registered prototypes."""
    module: NxModuleType = aml_finalize(vpm)
    assert any(cls.__name__ == "BaseSensor" for cls in module.__prototypes__)
    assert any(cls.__name__ == "Sensor" for cls in module.__prototypes__)

# endregion
