import anaximander.aml as nx


def test_classvar_data_parsers_and_validators(aml_module):
    module = aml_module("aml/value_parsers_module.py", "aml_value_parsers_module")
    sensor = module.Sensor
    assert sensor.metacharacters("merged").data["label"] == "OK"
