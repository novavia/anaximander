from tests.anaximander.aml.modules import type_hints_module as thm


def test_finalize_module_resolves_forward_refs(aml_finalize):
    aml_finalize(thm)
    field = thm.Sensor.declarators("merged").field["temperature"]
    assert field.type is thm.Temperature
    assert field.nullable is True
