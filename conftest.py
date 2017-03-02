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
import socket
import sys

from oauth2client.client import GoogleCredentials, \
    ApplicationDefaultCredentialsError

REMOTE_SERVER = "www.google.com"

ANAXIMANDER = os.path.dirname(__file__)
sys.path.append(ANAXIMANDER)

# =============================================================================
# Utility functions
# =============================================================================


def is_online():
    """Function that determines if the tester is online."""
    try:
        host = socket.gethostbyname(REMOTE_SERVER)
        socket.create_connection((host, 80), 2)
    except:
        return False
    else:
        return True

# =============================================================================
# Configuration
# =============================================================================


def pytest_addoption(parser):
    parser.addoption('--offline', action='store_true',
                     help="Skips tests that require an online connection.")


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
