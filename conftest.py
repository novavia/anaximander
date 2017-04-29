#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test configuration.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import os
import pytest
import sys

from oauth2client.client import GoogleCredentials, \
    ApplicationDefaultCredentialsError

# I believe this is done to enable testing in a docker container
# but I'm no longer sure. At any rate it is not harmful.
ANAXIMANDER = os.path.dirname(__file__)
sys.path.append(ANAXIMANDER)
from anaximander.utilities.functions import is_online

# =============================================================================
# Configuration
# =============================================================================


def pytest_addoption(parser):
    parser.addoption('--offline', action='store_true',
                     help="Skips tests that require an online connection.")
    parser.addoption('--bigtable', action='store_true',
                     help="Executes bigtable tests, which must be used \
                     sparingly because it triggers billing.")


def pytest_configure(config):
    """Sets configuration variables as environment variables."""
    global ONLINE
    try:
        offline_option = config.option.offline
    except AttributeError:
        offline_option = False
    ONLINE = not offline_option and is_online()
    os.environ['ONLINE'] = str(ONLINE)

    global GOOGLE_CREDENTIALS
    try:
        GoogleCredentials.get_application_default()
    except ApplicationDefaultCredentialsError:
        GOOGLE_CREDENTIALS = False
    else:
        GOOGLE_CREDENTIALS = True
    os.environ['GOOGLE_CREDENTIALS'] = str(GOOGLE_CREDENTIALS)

    global BIGTABLE
    try:
        BIGTABLE = config.option.bigtable
    except AttributeError:
        BIGTABLE = False
    os.environ['BIGTABLE'] = str(BIGTABLE)

# =============================================================================
# Pytest runner setup
# =============================================================================


def pytest_runtest_setup(item):
    if 'online' in item.keywords:
        if not ONLINE:
            pytest.skip("Tests are run offline.")
    if 'gcloud' in item.keywords:
        if not GOOGLE_CREDENTIALS:
            pytest.skip("No Google cloud credentials.")
    if 'bigtable' in item.keywords:
        if not BIGTABLE:
            pytest.skip("Bigtable option not selected.")
