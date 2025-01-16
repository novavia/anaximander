from anaximander.aml.metadescriptors.modeldescriptors import field
from anaximander.aml.prototypes.model import Model, compile


class derived_field(field):
    pass


@compile("test")
class C(Model):
    x: int = field()


class D(C):
    y: int = derived_field()


def test_compile():
    assert C.__compilations__ == {"test": {}}


def test_metadescriptors():
    assert C.metadescriptors() == {"x": C.x}
    assert D.metadescriptors() == {"x": C.x, "y": D.y}
    assert D.metadescriptors(inherited=False) == {"y": D.y}
    assert D.metadescriptors(derived_field) == {"y": D.y}
