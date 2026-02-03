# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2021–2022 Novavia Solutions, LLC

"""Tests configuration."""

# =============================================================================
# Imports and constants
# =============================================================================

import sys
from pathlib import Path

import pytest

TEST_DIRECTORY = Path(__file__).parent
REPO = TEST_DIRECTORY.parent
sys.path.append(str(REPO))
from anaximander.utilities.functions import is_online


CLOUD = False  # Stores whether to run @cloud tests

# =============================================================================
# Configuration
# =============================================================================


def pytest_addoption(parser):
    parser.addoption(
        "--cloud",
        action="store_true",
        help="Includes tests that interact with cloud resources.",
    )


def pytest_configure(config):
    """Sets configuration variables as environment variables."""
    config.addinivalue_line(
        "markers", "cloud: mark test to run only with the --cloud flag invoked"
    )
    config.addinivalue_line("filterwarnings", "ignore:.*cmp.*:DeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore:.*imp module:DeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore:Tasks:UserWarning")
    global CLOUD
    try:
        cloud_option = config.option.cloud
    except AttributeError:
        cloud_option = False
    if cloud_option:
        if not is_online():
            msg = "Cannot run cloud tests because the machine is offline."
            raise RuntimeError(msg)
        CLOUD = True


# =============================================================================
# Global fixtures
# =============================================================================


# =============================================================================
# Pytest runner setup
# =============================================================================


def pytest_runtest_setup(item):
    if "cloud" in item.keywords:
        if not CLOUD:
            pytest.skip("Cloud tests are skipped.")
