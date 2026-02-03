# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright © 2021–2022 Novavia Solutions, LLC

"""Framework initialization file."""

# =========================================================================== #
#                                   Imports                                   #
# =========================================================================== #

import os
from pathlib import Path

import dotenv

from .dataobjects import *
from .descriptors import *
from .meta import *
from .utilities.functions import boolean

# =========================================================================== #
#                            environment variables                            #
# =========================================================================== #

dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())

# =========================================================================== #
#                                   Globals                                   #
# =========================================================================== #

# Useful local directory shortcuts
CODEDIR = Path(os.path.dirname(os.path.abspath(__file__)))  # Code directory
REPO = CODEDIR.parent  # Git repository directory
DATAPATH = REPO / "data"  # Local data directory
WORKPATH = REPO / "work"  # Work directory
CACHEPATH = WORKPATH / ".cache"  # Document cache
TESTDIR = REPO / "tests"  # Test directory
TESTDATA = TESTDIR / "data"  # Test data directory
TESTWORK = TESTDIR / "work"  # Test work directory

# =========================================================================== #
#                             Environment settings                            #
# =========================================================================== #

# Interactive and Offline flags to manage imports
# The default values can get overriden by a caller script.
INTERACTIVE = boolean(os.environ.setdefault("INTERACTIVE", "True"))

# Storage flag signals whether the application works locally or in the cloud
STORAGE = os.environ.setdefault("STORAGE", "local")
