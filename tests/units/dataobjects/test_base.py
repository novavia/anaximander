from anaximander.descriptors.datatypes import nxdata

from anaximander.dataobjects import Data


def test_data_object_base(WeirdInteger, PositiveFloat):
    assert WeirdInteger.dataspec.pytype is int
    assert list(WeirdInteger.options) == ["parse", "conform", "validate", "integrate"]
    assert WeirdInteger.validate
    subtype = Data.subtype(dataspec=WeirdInteger.dataspec)
    assert subtype is WeirdInteger
    assert WeirdInteger.__kwargs__ == {
        "dataspec",
        "dataindex",
        "parse",
        "conform",
        "validate",
        "integrate",
    }
    assert issubclass(Data, nxdata)
    assert issubclass(Data.Base, nxdata)
    assert issubclass(WeirdInteger, nxdata)
    assert issubclass(PositiveFloat, nxdata)
    assert (
        repr(PositiveFloat.typespec)
        == "<typespec:Data|dataspec=DataSpec(pytype=<class 'float'>, name='PositiveFloat', nullable=False)>"
    )
