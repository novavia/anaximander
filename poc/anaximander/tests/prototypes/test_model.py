from anaximander.aml.metadescriptors.modeldescriptors import Field, field
from anaximander.aml.prototypes.model import Model, compile


class DerivedField(Field):
    pass


@compile("test")
class C(Model):
    x: int = field()


class D(C):
    y: int = DerivedField()  # type: ignore


def test_compile():
    assert C.__compilations__ == {"test": {}}


def test_metadescriptors():
    assert C.metadescriptors() == {"x": C.x}
    assert D.metadescriptors() == {"x": C.x, "y": D.y}
    assert D.metadescriptors(inherited=False) == {"y": D.y}
    assert D.metadescriptors(DerivedField) == {"y": D.y}
