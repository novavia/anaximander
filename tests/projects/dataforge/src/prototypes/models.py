from enum import Enum

import anaximander as nx


class MachineType(Enum):
    MOTOR = "motor"
    PUMP = "pump"


@nx.compile("sqlalchemy")
class Account(nx.Model):
    """A customer account, which also maps to an application tenant."""

    name: str = nx.field(key=True)

    facilities: list["Facility"] = nx.query()
    groups: list["MachineGroup"] = nx.query()


@nx.compile("sqlalchemy")
class Facility(nx.Model):
    """A customer facility where machines are located."""

    account: Account = nx.parent(key=True)
    name: str = nx.field(key=True)

    groups: list["MachineGroup"] = nx.query()
    machines: list["Machine"] = nx.query()


@nx.compile("sqlalchemy")
class MachineGroup(nx.Model):
    """An (optional) hierchical grouping of machines."""

    account: Account = nx.parent(key=True)
    name: str = nx.field(key=True)
    description: str = nx.field()

    machines: list["Machine"] = nx.query()


@nx.compile("sqlalchemy")
class Machine(nx.Model):
    """A digital twin of an industrial machine."""

    name: str = nx.field(key=True)
    facility: Facility = nx.parent(key=True)
    group: MachineGroup = nx.parent(key=True)
    mtype: str = nx.field(index=True)
    spec: dict[str, nx.data] = nx.field()

    monitors: list["Monitor"] = nx.query()


@nx.compile("sqlalchemy")
class Monitor(nx.Model):
    machine: Machine = nx.parent(key=True)
    name: str = nx.field(key=True)
    description: str = nx.field()

    device: "MonitoringDevice" = nx.query()


@nx.compile("sqlalchemy")
class MonitoringDevice(nx.Model):
    monitor: Monitor = nx.parent(unique=True)
    mac_id: str = nx.field(key=True)
    hardware: dict[str, nx.data] = nx.field()
