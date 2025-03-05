import sys
from enum import Enum
from types import ModuleType

import pytest

import anaximander as nx
from anaximander.aml.meta import set_type_annotations
from anaximander.compilers.model_compilers import SQLAlchemyCompiler


class Color(Enum):
    red = "red"
    green = "green"
    blue = "blue"


@nx.compile("sqlalchemy")
class TestModel(nx.Model):
    a: int = nx.field()
    b: dict[str, str] = nx.field()
    c: Color = nx.field()


@pytest.fixture(scope="module")
def module() -> ModuleType:
    module_ = sys.modules[__name__]
    set_type_annotations(module_)
    return module_


@pytest.fixture(scope="module")
def sqla_comp(module: ModuleType) -> SQLAlchemyCompiler:
    return SQLAlchemyCompiler(module)


def test_sqlalchemy_field_descriptor(sqla_comp: SQLAlchemyCompiler):
    assert sqla_comp.descriptor(TestModel.a) == "a: Mapped[int] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.b) == "b: Mapped[dict] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.c) == "c: Mapped[Color] = mapped_column()"


@pytest.mark.parametrize("path", ["blank.py"])
def test_sqlalchemy_compilation(compilation_success, path):
    assert compilation_success(path, compilation="sqlalchemy")


@pytest.mark.parametrize("path", ["basic.py"])
def test_dataclass_compilation(compilation_success, path):
    assert compilation_success(path, compilation="dataclasses")
