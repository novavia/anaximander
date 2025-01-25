from dataclasses import dataclass

from pydantic import BaseModel

from anaximander.aml.modeltypes import Model, ModelType, model, modeltype


class MyModel(Model):
    x: int


class MyOtherModelImplementation(metaclass=ModelType):
    pass


@dataclass
class MyDataClass:
    x: int


class MyPydanticModel(BaseModel):
    x: int


class X:
    pass


def test_is_modeltype():
    assert isinstance(MyModel, modeltype)
    assert isinstance(MyOtherModelImplementation, modeltype)
    assert isinstance(MyDataClass, modeltype)
    assert isinstance(MyPydanticModel, modeltype)
    assert not isinstance(X, modeltype)
    model.register(X)
    assert isinstance(X, modeltype)


