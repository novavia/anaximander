from dataclasses import dataclass




@dataclass
class MyModel:
    x: int
    y: list[str]
    z: str | None

