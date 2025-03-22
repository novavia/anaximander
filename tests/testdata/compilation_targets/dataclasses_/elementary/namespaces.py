from dataclasses import dataclass, field

from .. import prototypes_

@dataclass
class M:
    a: int = field(default=prototypes_.X)
    b: int = field(default=prototypes_.x)
