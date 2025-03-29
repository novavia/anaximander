import ast

import pytest

from anaximander import Model
from anaximander.aml.modeldescriptors import Field
from anaximander.compilers.bases import ProjectCompiler

def test_prepare_module(project_compiler):
    path = "elementary/basic.py"
    compiler: ProjectCompiler = project_compiler(path)
    module = compiler.modules[0]
    MyModel: type[Model] = getattr(module, "MyModel")
    assert module.__prototoypes__ == [MyModel]
    assert isinstance(MyModel.__ast__, ast.ClassDef)
    assert list(MyModel.metadescriptors()) == ["a", "b"]
    a: Field = getattr(MyModel, "a")
    assert isinstance(a, Field)
    assert a.name == "a"
    assert a.annotation == "int"
    assert a.hint is int
    assert isinstance(a.__ast__, ast.AnnAssign)

    path = "errors/invalid_bases.py"
    compiler: ProjectCompiler = project_compiler(path)
    with pytest.raises(TypeError):
        compiler._prepare_modules()

    path = "errors/multiple_model_inheritance.py"
    compiler: ProjectCompiler = project_compiler(path)
    with pytest.raises(TypeError):
        compiler._prepare_modules()
