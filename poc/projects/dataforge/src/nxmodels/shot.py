from typing import Optional

import anaximander as nx


@nx.compile("sqlalchemy")
class Account(nx.Model):
    """A customer account, which also maps to an application tenant."""

    name: str = nx.field(key=True)

    facilities: list["Facility"] = nx.relationship()
    groups: list["MachineGroup"] = nx.relationship()


@nx.compile("sqlalchemy")
class Facility(nx.Model):
    """A customer facility where machines are located."""

    account: Account = nx.field(key=True)
    name: str = nx.field(key=True)

    groups: list["MachineGroup"] = nx.relationship()
    machines: list["Machine"] = nx.relationship()


@nx.compile("sqlalchemy")
class MachineGroup(nx.Model):
    """A (optional) hierchical grouping of machines."""

    account: Account = nx.field(key=True)
    name: str = nx.field(key=True)
    description: str = nx.field()

    machines: list["Machine"] = nx.relationship()


@nx.compile("sqlalchemy")
class Machine(nx.Model):
    """A digital twin of an industrial machine."""

    name: str = nx.field(key=True)
    facility: Facility = nx.field(key=True)
    group: MachineGroup | None = nx.field(key=True)
    mtype: str = nx.field(index=True)
    spec: dict = nx.field()

    monitors: list["Monitor"] = nx.relationship()


@nx.compile("sqlalchemy")
class Monitor(nx.Model):
    machine: Machine = nx.field(key=True)
    name: str = nx.field(key=True)
    description: str = nx.field()

    device: Optional["MonitoringDevice"] = nx.relationship()


@nx.compile("sqlalchemy")
class MonitoringDevice(nx.Model):
    monitor: Optional[Monitor] = nx.field(unique=True)
    mac_id: str = nx.field(key=True)
    hardware: dict = nx.field()
