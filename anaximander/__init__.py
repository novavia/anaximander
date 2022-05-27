"""Framework initialization file."""

# =========================================================================== #
#                                   Imports                                   #
# =========================================================================== #

import os
from pathlib import Path

import dotenv

from .utilities.functions import boolean
from .utilities.nxyaml import NxYAML, yaml
from .descriptors import *
from .meta import *
from .dataobjects import *

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
TESTDIR = REPO / "test"  # Test directory
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
