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
    name: Mapped[str] = mapped_column(String(50), unique=True)

    facilities: Mapped[List["Facility"]] = relationship(back_populates="account", cascade="all, delete-orphan")
    groups: Mapped[List["MachineGroup"]] = relationship(back_populates="account", cascade="all, delete-orphan")


class Facility(Base):
    """A customer facility where machines are located."""

    __tablename__ = "facilities"
    id: Mapped[str] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(50), default="")

    account: Mapped["Account"] = relationship(back_populates="facilities", lazy="joined")
    groups: Mapped[List["MachineGroup"]] = relationship(back_populates="account", cascade="all, delete-orphan")
    machines: Mapped[List["Machine"]] = relationship(back_populates="facility", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("account_id", "name"),)


class MachineGroup(Base):
    """A (optional) hierchical grouping of machines."""

    __tablename__ = "machine_groups"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text, default="")

    account: Mapped["Account"] = relationship(back_populates="groups", lazy="joined")
    machines: Mapped[List["Machine"]] = relationship(back_populates="group")

    __table_args__ = (UniqueConstraint("account_id", "name"),)


class Machine(Base):
    """A digital twin of an industrial machine."""

    __tablename__ = "machines"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    facility_id: Mapped[int] = mapped_column(ForeignKey("facilities.id", ondelete="CASCADE"), index=True)
    group_id: Mapped[Optional[int]] = mapped_column(ForeignKey("machine_groups.id"), index=True)
    mtype: Mapped[str] = mapped_column(String(50), index=True)  # Machine type
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default={})  # Machine specifications

    facility: Mapped[Facility] = relationship(back_populates="machines", lazy="joined")
    group: Mapped[Optional[MachineGroup]] = relationship(back_populates="machines", lazy="joined")
    monitors: Mapped[List["Monitor"]] = relationship(back_populates="machine", cascade="all, delete-orphan")

    __table_args__ = (Index("machine_facility_group", facility_id, group_id, name, unique=True),)


class Monitor(Base):
    """A logical monitoring location on a machine."""

    __tablename__ = "monitors"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    machine_id: Mapped[int] = mapped_column(ForeignKey("machines.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[str] = mapped_column(Text, default="")

    machine: Mapped[Machine] = relationship(back_populates="monitors", lazy="joined")
    device: Mapped[Optional["MonitoringDevice"]] = relationship(back_populates="monitor", lazy="joined")

    __table_args__ = (UniqueConstraint("machine_id", "name"),)


class MonitoringDevice(Base):
    """A physical device enabling a machine monitoring function."""

    __tablename__ = "devices"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    mac_id: Mapped[str] = mapped_column(unique=True)
    monitor_id: Mapped[int] = mapped_column(ForeignKey("monitors.id"), unique=True, index=True)
    hardware: Mapped[dict[str, Any]] = mapped_column(JSON, default={})

    monitor: Mapped[Monitor] = relationship(back_populates="device", lazy="joined")
