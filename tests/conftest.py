# -*- coding: utf-8 -*-
"""
Pytest configuration, command-line options, markers, and shared fixtures.

Defines CLI flags to control test selection (offline, units, local/cloud deployment,
buggers), configures markers, and provides tempdata utilities and DB-related fixtures.
"""

# =============================================================================
# Imports and constants
# =============================================================================

import os
import shutil

# import sys
import filecmp
from pathlib import Path
from typing import Callable, Generator, Protocol

import pytest
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from anaximander.api.sqlalchemy_ import connections
# from anaximander.compilers import ProjectCompiler, ModuleCompiler
# from anaximander.projects import Project
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
    """Register custom command-line options.

    Args:
        parser (pytest.Parser): Pytest option parser to which flags are added.
    """
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
    """Initialize environment flags and register markers.

    Args:
        config (pytest.Config): Pytest configuration object.
    """
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

    config.addinivalue_line(
        "markers", "online: test is skipped if machine is not online"
    )
    config.addinivalue_line(
        "markers", "integration: test is skipped if only unit tests are run"
    )
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
    """Apply selection logic based on flags and markers before each test.

    Args:
        item (pytest.Item): The collected test item.
    """
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
# Custom fixtures
# =============================================================================


def reset_tempdata():
    """Reset the tempdata directory to a clean state and write a .gitignore."""
    shutil.rmtree(TEMPDATA, ignore_errors=True)
    TEMPDATA.mkdir(parents=True, exist_ok=True)
    gitignore = "\n".join(["*", "!.gitignore"])
    (TEMPDATA / ".gitignore").write_text(gitignore)


@pytest.fixture(scope="session", autouse=True)
def tempdata() -> Generator[Path, None, None]:
    """Provide a session-scoped tempdata directory.

    Yields:
        Path: The tempdata path, cleaned before and after the test session.
    """
    reset_tempdata()
    yield TEMPDATA
    reset_tempdata()


def make_tempdata_path(path: Path | str) -> Path:
    """Create a copy of a test resource under tempdata.

    If path is absolute, it must reside within the testdata directory.

    Args:
        path (Path | str): Relative path within testdata or an absolute path under testdata.

    Returns:
        Path: The corresponding path within tempdata.

    Raises:
        ValueError: If an absolute path is not within testdata.
    """
    path = Path(path)
    if path.is_absolute():
        if not path.is_relative_to(TESTDATA):
            msg = (
                "Only files and directories in the testdata folder can be fixtureized."
            )
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
    """Remove a path from tempdata if it exists.

    Args:
        path (Path | str): Path to a file or directory under tempdata.
    """
    path = Path(path)
    if not path.is_absolute():
        return
    if not path.is_relative_to(TEMPDATA):
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture(scope="function")
def funcpath() -> Generator[Callable[[Path], Path], None, None]:
    """Create function-scoped copies of resources into tempdata.

    Returns a callable that, when given a testdata path, produces a per-test copy
    under tempdata and tracks it for teardown.

    Yields:
        Callable[[Path | str], Path]: Copier function returning the tempdata path.
    """
    tempdata_paths = []

    def _path(path: Path | str) -> Path:
        tempdata_path = make_tempdata_path(path)
        tempdata_paths.append(tempdata_path)
        return tempdata_path

    yield _path

    for tempdata_path in tempdata_paths:
        teardown_tempdata_path(tempdata_path)


@pytest.fixture(scope="module")
def modpath() -> Generator[Callable[[Path], Path], None, None]:
    """Create module-scoped copies of resources into tempdata.

    Returns a callable that, when given a testdata path, produces a per-module copy
    under tempdata and tracks it for teardown.

    Yields:
        Callable[[Path | str], Path]: Copier function returning the tempdata path.
    """
    tempdata_paths = []

    def _path(path: Path | str) -> Path:
        tempdata_path = make_tempdata_path(path)
        tempdata_paths.append(tempdata_path)
        return tempdata_path

    yield _path

    for tempdata_path in tempdata_paths:
        teardown_tempdata_path(tempdata_path)


# @pytest.fixture
# def project() -> Generator[Callable[[Path], Project], None, None]:
#     """Creates a temporary project in tempdata for use in tests.

#     The path argument is a relative path in the testdata/prototypes directory pointing to either
#     a prototypes python file or a directory thereof. The name of the file or directory
#     is used as the project name, and the project is created under tempdata/projects.
#     """
#     project_paths = []

#     def _project(path: Path | str) -> Project:
#         path = TESTDATA / "prototypes" / Path(path)
#         project_name = path.stem
#         project = Project.create(
#             project_name, parent_directory=TEMP_PROJECTS, prototypes=path
#         )
#         project_paths.append(project.path)
#         return project

#     yield _project

#     for project_path in project_paths:
#         teardown_tempdata_path(project_path)


# class PathToProjectCompiler(Protocol):
#     def __call__(self, path: Path, *compilations: str) -> ProjectCompiler: ...


# @pytest.fixture
# def project_compiler(project) -> Generator[PathToProjectCompiler, None, None]:
#     """Returns a project compiler, optionally limited to specified handles.

#     The path argument is a relative path in the testdata/prototypes directory pointing to either
#     a prototypes python file or a directory thereof. The name of the file or directory
#     is used as the project name, and the project is created under tempdata/projects.
#     """

#     def _compiler(path: Path | str, *compilations: str) -> ProjectCompiler:
#         test_project = project(path)
#         compiler = ProjectCompiler(test_project, *compilations)
#         return compiler

#     yield _compiler


# class PathToModuleCompiler(Protocol):
#     def __call__(self, path: Path, compilation: str) -> ProjectCompiler: ...


# @pytest.fixture
# def module_compiler(project_compiler) -> Generator[PathToModuleCompiler, None, None]:
#     """Returns a module compiler.

#     The path argument is relative a path in the testdata/prototypes directory pointing to
#     a python module file.
#     """

#     def _compiler(path: Path | str, compilation: str) -> ModuleCompiler:
#         test_project_compiler: ProjectCompiler = project_compiler(path, compilation)
#         project = test_project_compiler.project
#         module_name = f"{project.slug}.api.prototypes_.__init__"
#         module_compiler = test_project_compiler.module_compiler(
#             module_name, compilation
#         )
#         return module_compiler

#     yield _compiler


# @pytest.fixture
# def compilation_success(
#     module_compiler,
# ) -> Generator[Callable[[Path, str], bool], None, None]:
#     """Returns an assertion of compilation success for a given module path and compilation handle."""

#     def _success(path: Path | str, compilation: str) -> bool:
#         path = Path(path)
#         test_module_compiler: ModuleCompiler = module_compiler(path, compilation)
#         test_module_compiler.run()
#         compilation_path = test_module_compiler.destination
#         target_path = TESTDATA / "compilation_targets" / (compilation + "_") / path
#         if target_path.exists():
#             return filecmp.cmp(compilation_path, target_path)
#         else:
#             return True

#     yield _success


@pytest.fixture
def engine(postgresql) -> Generator[Engine, None, None]:
    """Provide a SQLAlchemy Engine configured from the postgresql fixture.

    Args:
        postgresql: Fixture exposing connection parameters via .info.

    Yields:
        Engine: An engine connected to the test PostgreSQL instance.
    """
    params = KwargMap(postgresql.info, _=["host", "port", "dbname", "user", "password"])
    engine: Engine = connections.postgresql_engine(**params)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Generator[Session, None, None]:
    """Provide a SQLAlchemy ORM Session bound to the engine fixture.

    Args:
        engine (Engine): SQLAlchemy engine fixture.

    Yields:
        Session: A managed session bound to the provided engine.
    """
    with Session(engine) as session:
        yield session
    session.close()
