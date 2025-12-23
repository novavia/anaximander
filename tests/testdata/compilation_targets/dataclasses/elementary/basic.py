from dataclasses import dataclass, field


@dataclass
class MyModel:
    a: int = field()
    b: str = field()
