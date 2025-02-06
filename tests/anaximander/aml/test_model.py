from anaximander.aml.meta import compile
from anaximander.aml.model import Model
from anaximander.aml.modeldescriptors import Field, field


@compile("test")
class C(Model):
    x: int = field()


def test_compile():
    assert C.__compilations__ == {"test": {}}
    assert C.metadescriptors(Field) == {"x": C.x}


def test_set_annotation():
    C.__set_type_annotations__()
    assert C.x.annotation == "int"
    assert C.x.hint == int
