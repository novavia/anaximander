from anaximander import NXPATH, Project

TESTDIR = NXPATH / "tests"
TESTDATA = TESTDIR / "testdata"

SIMPLE_PROJECT_PATH = TESTDATA / "projects/simple_project"
COMPLEX_PROJECT_PATH = TESTDATA / "projects/complex_project"


def test_project_name():
    project = Project(path=SIMPLE_PROJECT_PATH)
    assert project.name == "simple_project"


def test_project_models_path():
    project = Project(path=SIMPLE_PROJECT_PATH)
    assert project.models_path == SIMPLE_PROJECT_PATH / "src/nxmodels"
    assert project.models_path.exists()


def test_project_compile_path():
    project = Project(path=SIMPLE_PROJECT_PATH)
    assert project.compile_path == SIMPLE_PROJECT_PATH / "src/simple_project"
    assert project.compile_path.exists()


def test_import_nxmodels_modules():
    simple_project = Project(path=SIMPLE_PROJECT_PATH)
    modules = simple_project.import_nxmodels_modules()
    assert len(modules) == 1
    assert modules[0].__name__ == "models"
    complex_project = Project(path=COMPLEX_PROJECT_PATH)
    modules = complex_project.import_nxmodels_modules()
    assert len(modules) == 10
    assert modules[0].__name__ == "package.subpackage.__init__"
    assert modules[-1].__name__ == "top_level_module"


def test_copy_nxmodels_modules():
    simple_project = Project(path=SIMPLE_PROJECT_PATH)
    simple_project.copy_nxmodels_modules()
    assert (simple_project.compile_path / "nxmodels_/models.py").exists()
    complex_project = Project(path=COMPLEX_PROJECT_PATH)
    complex_project.copy_nxmodels_modules()
    assert (complex_project.compile_path / "nxmodels_/package/subpackage/__init__.py").exists()
    assert (complex_project.compile_path / "nxmodels_/top_level_module.py").exists()
