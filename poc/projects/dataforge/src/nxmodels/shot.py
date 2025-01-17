from typing import Optional

import anaximander as nx


@nx.compile("sqlalchemy")
class Account(nx.Model):
    """A customer account, which also maps to an application tenant."""

    name: str = nx.Field(key=True)

    facilities: list["Facility"] = nx.relationship()
    groups: list["MachineGroup"] = nx.relationship()


@nx.compile("sqlalchemy")
class Facility(nx.Model):
    """A customer facility where machines are located."""

    account: Account = nx.Field(key=True)
    name: str = nx.Field(key=True)

    groups: list["MachineGroup"] = nx.relationship()
    machines: list["Machine"] = nx.relationship()


@nx.compile("sqlalchemy")
class MachineGroup(nx.Model):
    """A (optional) hierchical grouping of machines."""

    account: Account = nx.Field(key=True)
    name: str = nx.Field(key=True)
    description: str = nx.Field()

    machines: list["Machine"] = nx.relationship()


@nx.compile("sqlalchemy")
class Machine(nx.Model):
    """A digital twin of an industrial machine."""

    name: str = nx.Field(key=True)
    facility: Facility = nx.Field(key=True)
    group: MachineGroup | None = nx.Field(key=True)
    mtype: str = nx.Field(index=True)
    spec: dict = nx.Field()

    monitors: list["Monitor"] = nx.relationship()


@nx.compile("sqlalchemy")
class Monitor(nx.Model):
    machine: Machine = nx.Field(key=True)
    name: str = nx.Field(key=True)
    description: str = nx.Field()

    device: Optional["MonitoringDevice"] = nx.relationship()


@nx.compile("sqlalchemy")
class MonitoringDevice(nx.Model):
    monitor: Optional[Monitor] = nx.Field(unique=True)
    mac_id: str = nx.Field(key=True)
    hardware: dict = nx.Field()
