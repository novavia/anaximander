import anaximander.aml as nx


def test_minimal_aml_declarations():
    class Temperature(nx.Measurement):
        nx.metadata.unit = "C"

    class Sensor(nx.Model):
        temperature: Temperature = nx.data()

    assert Temperature.metacharacters("merged").metadata["unit"] == "C"
    assert "temperature" in Sensor.metacharacters("merged").field
