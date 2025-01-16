import pytest

from anaximander.aml.metadescriptors.base import metadescriptor

class C:
    x: int = metadescriptor()


class restricted_metadescriptor(metadescriptor):
    __reserved_names__ = ["x"]


def test_set_name():
    assert metadescriptor.__reserved_names__ == set()
    assert restricted_metadescriptor.__reserved_names__ == {"x"}
    assert C.x.name == "x"
    with pytest.raises(ValueError):
        class D:
            x: int = restricted_metadescriptor()
