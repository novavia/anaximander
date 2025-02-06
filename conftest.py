# -*- coding: utf-8 -*-
"""
Test configuration.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import os
import sys

import pytest

# I believe this is done to enable testing in a docker container
# but I'm no longer sure. At any rate it is not harmful.
REPO = os.path.dirname(__file__)
sys.path.append(REPO)

from anaximander.utils.funcs import boolean, offline  # noqa

# =============================================================================
# Configuration
# =============================================================================


def pytest_addoption(parser):
    parser.addoption(
        "--offline",
        action="store_true",
        help="Skips tests that require an online connection.",
    )
    parser.addoption(
        "--units",
        "--unit",
        action="store_true",
        help="Skips integration tests, leaving only unit tests.",
    )
    parser.addoption(
        "--local_deployment",
        "--local",
        action="store_true",
        help="Runs deployment tests that run locally, and excludes other tests.",
    )
    parser.addoption(
        "--cloud_deployment",
        "--cloud",
        action="store_true",
        help="Runs deployment tests in the cloud, and excludes other tests.",
    )
    parser.addoption(
        "--buggers",
        action="store_true",
        help="Only runs tests marked as 'buggers' (mark.bugger).",
    )


def pytest_configure(config):
    """Sets configuration variables as environment variables."""
    global OFFLINE
    global UNITS
    global LOCAL_DEPLOYMENT
    global CLOUD_DEPLOYMENT
    global DEBUG
    try:
        offline_option = config.option.offline
    except AttributeError:
        offline_option = False
    if offline_option:
        OFFLINE = offline(True)
    else:
        OFFLINE = offline()
    try:
        UNITS = config.option.units
    except AttributeError:
        UNITS = False
    try:
        LOCAL_DEPLOYMENT = config.option.local_deployment
    except AttributeError:
        LOCAL_DEPLOYMENT = False
    try:
        CLOUD_DEPLOYMENT = config.option.cloud_deployment
    except AttributeError:
        CLOUD_DEPLOYMENT = False
    try:
        DEBUG = config.option.buggers
    except AttributeError:
        DEBUG = False

    os.environ["UNITS"] = str(UNITS)
    os.environ["LOCAL_DEPLOYMENT"] = str(LOCAL_DEPLOYMENT)
    os.environ["CLOUD_DEPLOYMENT"] = str(CLOUD_DEPLOYMENT)

    global INTERACTIVE
    INTERACTIVE = boolean(os.environ.setdefault("INTERACTIVE", "False"))

    config.addinivalue_line("markers", "online: test is skipped if machine is not online")
    config.addinivalue_line("markers", "integration: test is skipped if only unit tests are run")
    config.addinivalue_line(
        "markers",
        "local_deployment: test is only run if the local_deployment flag is passed",
    )
    config.addinivalue_line(
        "markers",
        "cloud_deployment: test is only run if the cloud_deployment flag is passed",
    )
    config.addinivalue_line(
        "markers",
        "bugger: test is only run if the buggers flag is passed",
    )


# =============================================================================
# Pytest runner setup
# =============================================================================


def pytest_runtest_setup(item):
    if LOCAL_DEPLOYMENT:
        if "local_deployment" not in item.keywords:
            pytest.skip("Only local deployment tests are running.")
    else:
        if "local_deployment" in item.keywords:
            pytest.skip("Local deployment tests are excluded.")
    if CLOUD_DEPLOYMENT:
        if "cloud_deployment" not in item.keywords:
            pytest.skip("Only cloud deployment tests are running.")
    else:
        if "cloud_deployment" in item.keywords:
            pytest.skip("Cloud deployment tests are excluded.")
    if "online" in item.keywords:
        if OFFLINE:
            pytest.skip("Tests are run offline.")
    if "integration" in item.keywords:
        if UNITS:
            pytest.skip("Only unit tests are running.")
    if DEBUG:
        if "bugger" not in item.keywords:
            pytest.skip("Only bugger tests are running.")


# =============================================================================
# Debug configuration
# =============================================================================


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value
