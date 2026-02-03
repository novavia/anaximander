# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise AML declarative statements and binding syntaxes."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import anaximander.aml as nx

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_minimal_aml_declarations():
    """Validate minimal AML declarations compile and bind."""

    class Temperature(nx.Measurement):
        nx.metadata.unit = "C"

    class Sensor(nx.Model):
        temperature: Temperature = nx.data()

    assert Temperature.bindings("merged").metadata["unit"] == "C"
    assert "temperature" in Sensor.declarators("merged").field


def test_handle_binding_syntaxes():
    """Validate handle binding syntaxes on metadata."""

    class Temperature(nx.Measurement):
        nx.metadata["unit"] = "K"

    assert Temperature.bindings("merged").metadata["unit"] == "K"

# endregion
