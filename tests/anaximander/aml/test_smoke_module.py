from anaximander.aml.modules import NxModuleType
from tests.anaximander.aml.modules import value_parsers_module as vpm


def test_smoke_finalize_module(aml_finalize):
    module: NxModuleType = aml_finalize(vpm)
    assert any(cls.__name__ == "BaseSensor" for cls in module.__prototypes__)
    assert any(cls.__name__ == "Sensor" for cls in module.__prototypes__)
