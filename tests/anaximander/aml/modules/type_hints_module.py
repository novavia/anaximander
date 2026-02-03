# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Provide a test module for type hint resolution."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import anaximander.aml as nx

# endregion

# =============================================================================
# Test models
# =============================================================================
# region Test models


class Temperature(nx.Measurement):
    """Test measurement type for forward-reference resolution."""
    pass


class Sensor(nx.Model):
    """Test model with forward-referenced measurement field."""
    temperature: Temperature | None = nx.data()

# endregion
