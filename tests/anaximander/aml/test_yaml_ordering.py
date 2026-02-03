# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise declarator ordering in AML YAML serialization."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

from __future__ import annotations

from anaximander.aml.declarators import (
    BackLinkProtodescriptor,
    DataProtodescriptor,
    LinkProtodescriptor,
)

# endregion

# =============================================================================
# Helpers
# =============================================================================
# region Helpers


def _keys(obj) -> list[str]:
    """Return ordered serialization keys for a declarator."""
    return list(obj.to_dict().keys())

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_data_protodescriptor_ordering():
    """Ensure data protodescriptor ordering matches expected YAML serialization."""
    expected = [
        "name",
        "owner",
        "ordinal",
        "annotation",
        "hint",
        "type",
        "nullable",
        "classvar",
        "default",
        "factory",
        "typekey",
        "required",
        "load",
        "unique",
        "index",
        "key",
        "sequence",
        "timestamp",
        "start_time",
        "end_time",
        "period",
        "location",
        "geom",
        "validator",
        "gt",
        "ge",
        "lt",
        "le",
        "min_length",
        "max_length",
        "pattern",
        "repr",
        "doc",
        "config",
    ]
    got = _keys(DataProtodescriptor())
    assert got[: len(expected)] == expected


def test_link_protodescriptor_ordering():
    """Ensure link protodescriptor ordering matches expected YAML serialization."""
    expected = [
        "name",
        "owner",
        "ordinal",
        "annotation",
        "hint",
        "type",
        "nullable",
        "classvar",
        "default",
        "factory",
        "required",
        "load",
        "unique",
        "key",
        "on_delete",
        "validator",
        "repr",
        "doc",
        "config",
    ]
    got = _keys(LinkProtodescriptor())
    assert got[: len(expected)] == expected


def test_backlink_protodescriptor_ordering():
    """Ensure backlink protodescriptor ordering matches expected YAML serialization."""
    expected = [
        "name",
        "owner",
        "ordinal",
        "annotation",
        "hint",
        "type",
        "nullable",
        "classvar",
        "load",
        "unique",
        "via",
        "limit",
        "repr",
        "doc",
        "config",
    ]
    got = _keys(BackLinkProtodescriptor())
    assert got[: len(expected)] == expected

# endregion
