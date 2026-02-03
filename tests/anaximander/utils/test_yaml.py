# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise YAML round-trip helpers."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from anaximander.utils.yaml import nx_yaml_dump, nx_yaml_load

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_nx_yaml_roundtrip_basic():
    """Round-trip a basic payload through NX YAML helpers."""
    payload = {"value": "ok", "num": 3, "flag": False, "none": None}
    dumped = nx_yaml_dump(payload)
    loaded = nx_yaml_load(dumped)
    assert loaded == payload

# endregion
