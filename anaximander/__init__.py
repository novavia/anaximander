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
OFFLINE = boolean(os.environ.setdefault('OFFLINE', str(not is_online())))


from . import utilities
from . import registries
from . import meta
from . import data


# Sets the anaximander directory
NXDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Appends the test directory to the import path
TESTDIR = os.path.join(NXDIR, 'tests')
sys.path.append(TESTDIR)

