import shutil
from typing import Generator

import pytest

from anaximander import REPO, Project
from tests.utils import fixturize

TESTDIR = REPO / "tests"
TESTDATA = TESTDIR / "testdata"
TEMPDATA = TESTDIR / "tempdata"

SIMPLE_PROJECT_PATH = TESTDATA / "projects/simple_project"
COMPLEX_PROJECT_PATH = TESTDATA / "projects/complex_project"
BAD_IMPORTS_PROJECT_PATH = TESTDATA / "projects/bad_imports"


@pytest.fixture
def simple_project() -> Generator[Project, None, None]:
    with fixturize(SIMPLE_PROJECT_PATH) as path:
        yield Project(path)


@pytest.fixture
def complex_project() -> Generator[Project, None, None]:
    with fixturize(COMPLEX_PROJECT_PATH) as path:
        yield Project(path)


@pytest.fixture
def bad_imports_project() -> Generator[Project, None, None]:
    with fixturize(BAD_IMPORTS_PROJECT_PATH) as path:
        yield Project(path)


def test_create_project():
    project_name = "test"
    Project.create(project_name, root_directory=TEMPDATA / "projects")
    assert (TEMPDATA / "projects" / project_name).exists()
    assert (TEMPDATA / "projects" / project_name / "src").exists()
    shutil.rmtree(TEMPDATA / "projects" / project_name)


def test_project_name(simple_project: Project):
    assert simple_project.name == "simple_project"


def test_project_models_path(simple_project: Project):
    assert simple_project.models_path == TEMPDATA / "projects/simple_project/src/nxmodels"
    assert simple_project.models_path.exists()


def test_project_compile_path(simple_project: Project):
    assert simple_project.compile_path == TEMPDATA / "projects/simple_project/src/simple_project"
    assert simple_project.compile_path.exists()


def test_copy_nxmodels_modules(simple_project: Project, complex_project: Project):
    simple_project.copy_nxmodels_modules()
    assert (simple_project.compile_path / "nxmodels_/models.py").exists()
    complex_project.copy_nxmodels_modules()
    assert (complex_project.compile_path / "nxmodels_/package/subpackage/__init__.py").exists()
    assert (complex_project.compile_path / "nxmodels_/top_level_module.py").exists()


def test_import_nxmodels_modules(
    simple_project: Project, complex_project: Project, bad_imports_project: Project
):
    simple_project.copy_nxmodels_modules()
    modules = simple_project.import_nxmodels_modules()
    assert len(modules) == 1
    assert modules[0].__name__ == "models"

    complex_project.copy_nxmodels_modules()
    modules = complex_project.import_nxmodels_modules()
    assert len(modules) == 10
    assert modules[0].__name__ == "package.subpackage.__init__"
    assert modules[-1].__name__ == "top_level_module"

    bad_imports_project.copy_nxmodels_modules()
    with pytest.raises(ImportError):
        bad_imports_project.import_nxmodels_modules()
