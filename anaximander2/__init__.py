#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Init module to the Anaximander framework.

This module is part of the Anaximander project.
Copyright (C) Novavia Solutions, LLC.
"""

import os.path
import sys


from .utilities.functions import boolean, is_online

# Tests whether running locally or on Google Cloud
IS_LOCAL = not('SERVER_SOFTWARE' in os.environ)
LOCAL = boolean(os.environ.setdefault('LOCAL', str(IS_LOCAL)))


# Interactive and Offline flags to manage imports
# The default values can get overriden by a caller script.
INTERACTIVE = boolean(os.environ.setdefault('INTERACTIVE', 'True'))


def offline(assertion=None):
    """Returns application status, or sets it if assertion is passed.

    Asserting offline is False is subject to verifying that the application
    truly is online.
    """
    if assertion is not None:
        if assertion is False:
            try:
                assert is_online()
            except AssertionError:
                pass
            else:
                os.environ['OFFLINE'] = 'False'
                return False
        os.environ['OFFLINE'] = 'True'
        return True
    try:
        return boolean(os.environ['OFFLINE'])
    except KeyError:
        which = not is_online()
        os.environ['OFFLINE'] = str(which)
        return which


from . import utilities


# Sets the anaximander directory
NXDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Appends the test directory to the import path
TESTDIR = os.path.join(NXDIR, 'tests')
sys.path.append(TESTDIR)
