# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Provide a test module for value parser and validator behavior."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from typing import ClassVar

import anaximander.aml as nx

# endregion

# =============================================================================
# Test models
# =============================================================================
# region Test models

class BaseSensor(nx.Model):
    """Base sensor model with classvar label data."""

    label: ClassVar[str] = nx.data()


class Sensor(BaseSensor):
    """Sensor model with parser and validator hooks."""

    label = "  ok  "

    @nx.parser("label")
    def _parse_label(cls, value: str) -> str:
        """Normalize the label value for test assertions."""
        return value.strip().upper()

    @nx.validator("label")
    def _validate_label(cls, value: str) -> bool:
        """Validate normalized label values."""
        return value.isupper()

# endregion
