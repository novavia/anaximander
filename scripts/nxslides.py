import anaximander as nx
from datetime import datetime, timedelta
from enum import Enum


@nx.archetype
class Measurement(nx.Float):
    unit: str = nx.metadata()


class Temperature(Measurement):
    unit = "C"


class MachineType(str, Enum):
    motor = "motor"
    pump = "pump"
    other = "other"


class Machine(nx.Entity):
    id: int = nx.data(nxid=True)
    machine_type: MachineType = nx.data()
    machine_name: str = nx.data()
    group: "MachineGroup" = nx.data()
    nameplate: "Nameplate" = nx.data()
    operating_range: "OperatingRange" = nx.data()
    temperature_probe: "TemperatureProbe" = nx.relation(
        "TemperatureProbe", backref="machine"
    )


class MachineGroup(nx.Entity):
    id: int = nx.data(nxid=True)
    group_name: str = nx.data()
    machines: nx.Set[Machine] = nx.relation(Machine, backref="group")


class TemperatureProbe(nx.Entity):
    id: int = nx.data(nxid=True)
    machine: Machine = nx.data()


class TemperatureSample(nx.Sample):
    nxperiod = timedelta(seconds=30)  # sets metadata
    probe: TemperatureProbe = nx.data(nxkey=True)
    time: datetime = nx.data(nxtime=True)
    temperature: Temperature = nx.data()


class DailyTemperatureJournal(nx.Journal):
    nxperiod = timedelta(days=1)
    probe: TemperatureProbe = nx.data()
    day: datetime = nx.data(nxtime=True)
    min_temperature: Temperature = nx.data()
    avg_temperature: Temperature = nx.data()
    max_temperature: Temperature = nx.data()


class OverheatIncident(nx.Session):
    machine: Machine = nx.data(nxkey=True)
    start: datetime = nx.data(nxstart=True)
    end: datetime = nx.data(nxend=True)
    report: "IncidentReport" = nx.data()


class Nameplate(nx.Spec):
    make: str = nx.data()
    model: str = nx.data()
    serial_number: str = nx.data(regex="\w{9,17}")


class OperatingRange(nx.Spec):
    min_temp: Temperature = nx.data()
    max_temp: Temperature = nx.data()
    alert_buffer: timedelta = nx.data()


class IncidentReport(nx.Spec):
    stats: nx.Spec = nx.data()
    response_time: timedelta = nx.data()
