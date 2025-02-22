# -*- coding: utf-8 -*-
"""
Test configuration.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import os
import shutil

# import sys
from pathlib import Path
from typing import Callable, Generator

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from anaximander.api.sqlalchemy_ import connections
from anaximander.projects import Project
from anaximander.utils import KwargMap
from anaximander.utils.funcs import boolean, offline  # noqa

TESTDIR = Path(__file__).parent
REPO = TESTDIR.parent
TESTDATA = TESTDIR / "testdata"
TEMPDATA = TESTDIR / "tempdata"
TEMP_PROJECTS = TEMPDATA / "projects"

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


def pytest_runtest_setup(item):  # noqa: C901
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


# =============================================================================
# Custom fixtures
# =============================================================================


def reset_tempdata():
    shutil.rmtree(TEMPDATA, ignore_errors=True)
    TEMPDATA.mkdir(parents=True, exist_ok=True)
    gitignore = "\n".join(["*", "!.gitignore"])
    (TEMPDATA / ".gitignore").write_text(gitignore)


@pytest.fixture(scope="session", autouse=True)
def tempdata() -> Generator[Path, None, None]:
    reset_tempdata()
    yield TEMPDATA
    reset_tempdata()


def make_tempdata_path(path: Path | str) -> Path:
    """Creates a path in tempdata for use in tests."""
    path = Path(path)
    if path.is_absolute():
        if not path.is_relative_to(TESTDATA):
            msg = "Only files and directories in the testdata folder can be fixtureized."
            raise ValueError(msg)
        relative_path = path.relative_to(TESTDATA)
    else:
        relative_path = path
    source = TESTDATA / relative_path
    destination = TEMPDATA / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_file():
        source.copy(destination)
    else:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def teardown_tempdata_path(path: Path | str):
    """Teardown for tempdata fixtures."""
    path = Path(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture(scope="function")
def function_data() -> Generator[Callable[[Path], Path], None, None]:
    """Creates a temporary copy of a file or directory into tempdata for use in tests."""
    tempdata_path = Path()  # This is a placeholder

    def _path(path: Path | str) -> Path:
        nonlocal tempdata_path
        tempdata_path = make_tempdata_path(path)
        return tempdata_path

    yield _path

    # teardown_tempdata_path(tempdata_path)


@pytest.fixture(scope="module")
def module_data() -> Generator[Callable[[Path], Path], None, None]:
    """Creates a temporary copy of a file or directory into tempdata for use in tests."""
    tempdata_path = Path()  # This is a placeholder

    def _path(path: Path | str) -> Path:
        nonlocal tempdata_path
        tempdata_path = make_tempdata_path(path)
        return tempdata_path

    yield _path

    # teardown_tempdata_path(tempdata_path)


@pytest.fixture
def project(function_data) -> Generator[Callable[[Path], Project], None, None]:
    """Creates a temporary project in tempdata for use in tests.

    The path argument is a path in the testdata/prototypes directory pointing to either
    a prototypes python file or a directory thereof. The name of the file or directory
    is used as the project name, and the project is created under tempdata/projects.
    """
    project_path = Path()  # This is a placeholder

    def _project(path: Path | str) -> Project:
        path = TESTDATA / "prototypes" / path
        project_name = path.stem
        project = Project.create(project_name, parent_directory=TEMP_PROJECTS, prototypes=path)
        nonlocal project_path
        project_path = project.path
        return project

    yield _project

    # teardown_tempdata_path(project_path)


@pytest.fixture
def engine(postgresql) -> Generator[Engine, None, None]:
    """A SQLAlchemy engine fixture."""
    params = KwargMap(postgresql.info, _=["host", "port", "dbname", "user", "password"])
    engine: Engine = connections.postgresql_engine(**params)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Generator[Session, None, None]:
    """A SQLAlchemy session fixture."""
    with Session(engine) as session:
        yield session
    session.close()
