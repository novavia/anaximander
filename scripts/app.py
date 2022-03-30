from datetime import datetime
from typing import Optional

import anaximander as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class Machine(nx.Entity):
    id: int = nx.data(nx_id=True)
    machine_type: str = nx.data()
    machine_floor: Optional[str] = nx.data()


class Temperature(nx.Measurement):
    unit = "Celsius"
    ge = -273
    le = 1000


class TemperatureSample(nx.Journal):
    machine_id: int = nx.key()
    timestamp: datetime = nx.timestamp()
    temperature: Temperature = nx.data(ge=0)

    @timestamp.parser
    def to_datetime(cls, value):
        return pd.to_datetime(value).to_pydatetime()


m0 = Machine(id=0, machine_type="motor")

times = pd.date_range(start="2022-2-18 12:00", freq="s", periods=10)
temperatures = [45.0, 46.0, 45.0, 50.0, 52.0, 50.0, 48.0, 51.0, 47.0, 50.0]
log = nx.Table[TemperatureSample](
    dict(machine_id=0, timestamp=times, temperature=temperatures)
)
