import anaximander.aml as nx


def test_minimal_aml_declarations():
    class Temperature(nx.Measurement):
        nx.metadata.unit = "C"

    class Sensor(nx.Model):
        temperature: Temperature = nx.data()

    assert Temperature.bindings("merged").metadata["unit"] == "C"
    assert "temperature" in Sensor.declarators("merged").field


def test_interface_binding_syntaxes():
    class Temperature(nx.Measurement):
        nx.metadata["unit"] = "K"

    assert Temperature.bindings("merged").metadata["unit"] == "K"
