import pytest
from anaximander.aml.metadescriptors.base import Metadescriptor


class C:
    x: int = Metadescriptor()  # type: ignore


class restricted_metadescriptor(Metadescriptor):
    __reserved_names__ = ["x"]


def test_set_name():
    assert Metadescriptor.__reserved_names__ == set()
    assert restricted_metadescriptor.__reserved_names__ == {"x"}
    assert C.x.name == "x"  # type: ignore
    with pytest.raises(ValueError):

        class D:
            x: int = restricted_metadescriptor()  # type: ignore
