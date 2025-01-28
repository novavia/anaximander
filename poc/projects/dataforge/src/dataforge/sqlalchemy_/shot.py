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
    name: Mapped[] = mapped_column()

class Facility(Base):
    """A customer facility where machines are located."""

    __tablename__ = "facilitys"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account: Mapped[] = mapped_column()
    name: Mapped[] = mapped_column()

class MachineGroup(Base):
    """A (optional) hierchical grouping of machines."""

    __tablename__ = "machinegroups"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account: Mapped[] = mapped_column()
    name: Mapped[] = mapped_column()
    description: Mapped[] = mapped_column()

class Machine(Base):
    """A digital twin of an industrial machine."""

    __tablename__ = "machines"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[] = mapped_column()
    facility: Mapped[] = mapped_column()
    group: Mapped[] = mapped_column()
    mtype: Mapped[] = mapped_column()
    spec: Mapped[] = mapped_column()

class Monitor(Base):
    """None"""

    __tablename__ = "monitors"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    machine: Mapped[] = mapped_column()
    name: Mapped[] = mapped_column()
    description: Mapped[] = mapped_column()

class MonitoringDevice(Base):
    """None"""

    __tablename__ = "monitoringdevices"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    monitor: Mapped[] = mapped_column()
    mac_id: Mapped[] = mapped_column()
    hardware: Mapped[] = mapped_column()

