from dataclasses import dataclass

import pytest

from anaximander.descriptors.attributes import Attribute


def test_attribute():
    x = Attribute("x", retriever=lambda s: getattr(s, "_x"))
    y = Attribute("y", retriever=lambda s: getattr(s, "_y"), transformer=round)

    # Test on simple class

    class C:
        _x: float = 0.99
        _y: float = 0.99

    setattr(C, "x", x)
    setattr(C, "y", y)

    assert isinstance(C.x, Attribute)
    c = C()
    assert c.x == 0.99
    assert c.y == 1

    with pytest.raises(AttributeError):
        c.x = 1

    # Test with metaclass

    class Meta(type):
        _x: float = 0.01
        _y: float = 0.01

    setattr(Meta, "x", x)
    setattr(Meta, "y", y)

    @dataclass
    class C(metaclass=Meta):
        _x: float
        _y: float = 0.99

    assert C.x == 0.01
    assert C.y == 1

    # This doesn't raise because the supplied values are attributes
    setattr(C, "x", x)
    setattr(C, "y", y)

    # However the return values remain the same, driven by the metaclass

    assert C.x == 0.01
    assert C.y == 1

    c = C(1)
    assert c.x == 1
    assert c.y == 1

    c._y = 1.99
    assert C.y == 1
    assert c.y == 2
