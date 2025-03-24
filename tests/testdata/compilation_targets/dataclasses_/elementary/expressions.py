from dataclasses import dataclass, field
from datetime import datetime

DT_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class MyModel:
    t: datetime = field(default=datetime.strptime("2022-01-01 00:00:00", DT_FORMAT))
