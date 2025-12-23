from typing import Any, List, Optional

from sqlalchemy import (
    JSON,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from . import Base


class Account(Base):
    """A customer account, which also maps to an application tenant."""

    __tablename__ = "accounts"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column()
    facilities: Mapped[list['Facility']] = relationship()
    groups: Mapped[list['MachineGroup']] = relationship()


class Facility(Base):
    """A customer facility where machines are located."""

    __tablename__ = "facilities"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account: Mapped[Account] = mapped_column()
    name: Mapped[str] = mapped_column()
    groups: Mapped[list['MachineGroup']] = relationship()
    machines: Mapped[list['Machine']] = relationship()


class MachineGroup(Base):
    """An (optional) hierchical grouping of machines."""

    __tablename__ = "machine_groups"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account: Mapped[Account] = mapped_column()
    name: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column()
    machines: Mapped[list['Machine']] = relationship()


class Machine(Base):
    """A digital twin of an industrial machine."""

    __tablename__ = "machines"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column()
    facility: Mapped[Facility] = mapped_column()
    group: Mapped[MachineGroup] = mapped_column()
    mtype: Mapped[str] = mapped_column()
    spec: Mapped[dict] = mapped_column()
    monitors: Mapped[list['Monitor']] = relationship()


class Monitor(Base):
    __tablename__ = "monitors"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    machine: Mapped[Machine] = mapped_column()
    name: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column()
    device: Mapped[MonitoringDevice] = relationship()


class MonitoringDevice(Base):
    __tablename__ = "monitoring_devices"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    monitor: Mapped[Monitor] = mapped_column()
    mac_id: Mapped[str] = mapped_column()
    hardware: Mapped[dict] = mapped_column()

