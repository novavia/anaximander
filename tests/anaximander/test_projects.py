import pytest

from anaximander import REPO, Project

TESTDIR = REPO / "tests"
TESTDATA = TESTDIR / "testdata"
TEMPDATA = TESTDIR / "tempdata"

SIMPLE_PROJECT_PATH = TESTDATA / "projects/simple_project"
COMPLEX_PROJECT_PATH = TESTDATA / "projects/complex_project"
BAD_IMPORTS_PROJECT_PATH = TESTDATA / "projects/bad_imports"


def test_create_project(project):
    simple_project: Project = project("simple_project")
    assert simple_project.name == "simple_project"
    assert simple_project.config_path.exists()
    assert simple_project.prototypes_source_path.exists()
    assert simple_project.application_path.exists()


def test_copy_prototypes(project):
    simple_project: Project = project("simple_project")
    complex_project: Project = project("complex_project")
    simple_project.copy_prototypes()
    assert (simple_project.api_prototypes_path / "models.py").exists()
    complex_project.copy_prototypes()
    assert (complex_project.api_prototypes_path / "package/subpackage/__init__.py").exists()
    assert (complex_project.api_prototypes_path / "top_level_module.py").exists()


def test_import_prototypes(project):
    simple_project: Project = project("simple_project")
    complex_project: Project = project("complex_project")
    bad_imports_project: Project = project("bad_imports")

    simple_project.copy_prototypes()
    modules = simple_project.import_prototypes()
    assert len(modules) == 2
    assert set(m.__name__ for m in modules) == {"simple_project.api.prototypes_.models", 
                                                "simple_project.api.prototypes_.__init__"}

    complex_project.copy_prototypes()
    modules = complex_project.import_prototypes()
    assert len(modules) == 13
    assert modules[0].__name__ == "complex_project.api.prototypes_.package.subpackage.__init__"
    assert modules[-1].__name__ == "complex_project.api.prototypes_.top_level_module"

    bad_imports_project.copy_prototypes()
    with pytest.raises(ImportError):
        bad_imports_project.import_prototypes()
