from tests.anaximander.aml.modules import value_parsers_module as vpm


def test_classvar_data_parsers_and_validators(aml_finalize):
    aml_finalize(vpm)
    assert vpm.Sensor.bindings("merged").data["label"] == "OK"
