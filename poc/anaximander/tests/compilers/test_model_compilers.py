import sys
from enum import Enum
from types import ModuleType
from typing import Literal, Optional

import anaximander as nx
import pytest
from anaximander.aml.meta import set_type_annotations
from anaximander.compilers.model_compilers import SQLAlchemyCompiler


class Color(Enum):
    red = "red"
    green = "green"
    blue = "blue"


@nx.compile("sqlalchemy")
class TestModel(nx.Model):
    a: int = nx.field()
    b: str | None = nx.field()
    c: Optional[float] = nx.field()
    d: dict[str, str] = nx.field()
    e: float | str = nx.field()
    f: Literal["f", 0] = nx.field()
    g: Color = nx.field()


@nx.compile("sqlalchemy")
class InvalidFieldsModel(nx.Model):
    a: TestModel = nx.field()
    b: dict = nx.field()


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
    assert sqla_comp.descriptor(TestModel.b) == "b: Mapped[str | None] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.c) == "c: Mapped[Optional[float]] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.d) == "d: Mapped[dict[str, str]] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.e) == "e: Mapped[float | str] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.f) == "f: Mapped[Literal['f', 0]] = mapped_column()"
    assert sqla_comp.descriptor(TestModel.g) == "g: Mapped[Color] = mapped_column()"
    with pytest.raises(TypeError):
        sqla_comp.descriptor(InvalidFieldsModel.a)
    with pytest.raises(TypeError):
        sqla_comp.descriptor(InvalidFieldsModel.b)
