from dataclasses import dataclass, field


@dataclass
class A:
    a: int = field()

@dataclass
class B(A):
    b: int = field()
