from anaximander.utils.funcs import subclasses


class C0:
    pass


class C1(C0):
    pass


class D1(C0):
    pass


class D2(D1):
    pass


def test_subclasses():
    assert subclasses(C0) == [C1, D1, D2]
    assert subclasses(C0, strict=False) == [C0, C1, D1, D2]
    assert subclasses(C0, depth=0) == []
    assert subclasses(C0, depth=1) == [C1, D1]
    assert subclasses(C0, depth=2) == [C1, D1, D2]
    assert subclasses(C0, depth=3) == [C1, D1, D2]
