# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise value parser behavior for AML data bindings."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from tests.anaximander.aml.modules import value_parsers_module as vpm

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_classvar_data_parsers_and_validators(aml_finalize):
    """Apply parser/validator logic to classvar data bindings."""
    aml_finalize(vpm)
    assert vpm.Sensor.bindings("merged").data["label"] == "  ok  "

# endregion
