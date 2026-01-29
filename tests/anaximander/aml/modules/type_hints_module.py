from __future__ import annotations

import anaximander.aml as nx


class Temperature(nx.Measurement):
    pass


class Sensor(nx.Model):
    temperature: Temperature | None = nx.data()
