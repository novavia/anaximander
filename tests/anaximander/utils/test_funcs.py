# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2024–2026 Novavia Solutions, LLC

"""Exercise utility helpers for subclasses, naming, and module detection."""

# =============================================================================
# Imports
# =============================================================================
# region Imports

import sys

import anaximander as nx
from anaximander.utils.funcs import (
    camel_to_snake,
    is_package_init,
    pluralize,
    subclasses,
    type_name_to_collection_name,
)

# endregion

# =============================================================================
# Test scaffolding
# =============================================================================
# region Test scaffolding


class C0:
    """Base class for subclass enumeration tests."""


class C1(C0):
    """First-level subclass for enumeration tests."""


class D1(C0):
    """Sibling subclass for enumeration tests."""


class D2(D1):
    """Second-level subclass for enumeration tests."""

# endregion

# =============================================================================
# Tests
# =============================================================================
# region Tests


def test_subclasses():
    """Verify subclasses() respects strict inclusion and depth limits."""
    assert subclasses(C0) == [C1, D1, D2]
    assert subclasses(C0, strict=False) == [C0, C1, D1, D2]
    assert subclasses(C0, depth=0) == []
    assert subclasses(C0, depth=1) == [C1, D1]
    assert subclasses(C0, depth=2) == [C1, D1, D2]
    assert subclasses(C0, depth=3) == [C1, D1, D2]


def test_camel_to_snake():
    """Check camel_to_snake() handles mixed, leading-cap, and idempotent cases."""
    assert camel_to_snake("camelCase") == "camel_case"
    assert camel_to_snake("CamelCase") == "camel_case"
    assert camel_to_snake("camel_case") == "camel_case"
    assert camel_to_snake("getHTTPResponseCode") == "get_http_response_code"
    assert camel_to_snake("") == ""


def test_pluralize():
    """Ensure pluralize() pluralizes simple words and leaves others unchanged."""
    assert pluralize("word") == "words"
    assert pluralize("Words") == "Words"
    assert pluralize("index") == "indexes"


def test_type_name_to_collection_name():
    """Validate conversion from CamelCase to snake_case plural collection names."""
    assert type_name_to_collection_name("CamelCase") == "camel_cases"
    assert type_name_to_collection_name("HTTPResponseCode") == "http_response_codes"
    assert type_name_to_collection_name("CamelCaseThing") == "camel_case_things"


def test_is_package_init():
    """Confirm is_package_init distinguishes regular modules from package __init__."""
    this_module = sys.modules[__name__]
    assert not is_package_init(this_module)
    assert is_package_init(nx)

# endregion
