from typing import ClassVar

import anaximander.aml as nx


class BaseSensor(nx.Model):
    label: ClassVar[str] = nx.data()


class Sensor(BaseSensor):
    label = "  ok  "

    @nx.parser("label")
    def _parse_label(cls, value: str) -> str:
        return value.strip().upper()

    @nx.validator("label")
    def _validate_label(cls, value: str) -> bool:
        return value.isupper()
