from dataclasses import dataclass, field


@dataclass
class A:
    a: int = field()
    b: "B" = field()

@dataclass
class B:
    a: A = field()
    b: str = field()
