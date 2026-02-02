from __future__ import annotations

from anaximander.aml.declarators import BackLinkProtodescriptor, DataProtodescriptor, LinkProtodescriptor


def _keys(obj) -> list[str]:
    return list(obj.to_dict().keys())


def test_data_protodescriptor_ordering():
    expected = [
        "name",
        "owner",
        "ordinal",
        "annotation",
        "hint",
        "type",
        "nullable",
        "classvar",
        "default",
        "factory",
        "typekey",
        "required",
        "load",
        "unique",
        "index",
        "key",
        "sequence",
        "timestamp",
        "start_time",
        "end_time",
        "period",
        "location",
        "geom",
        "validator",
        "gt",
        "ge",
        "lt",
        "le",
        "min_length",
        "max_length",
        "pattern",
        "repr",
        "doc",
        "config",
    ]
    got = _keys(DataProtodescriptor())
    assert got[: len(expected)] == expected


def test_link_protodescriptor_ordering():
    expected = [
        "name",
        "owner",
        "ordinal",
        "annotation",
        "hint",
        "type",
        "nullable",
        "classvar",
        "default",
        "factory",
        "required",
        "load",
        "unique",
        "key",
        "on_delete",
        "validator",
        "repr",
        "doc",
        "config",
    ]
    got = _keys(LinkProtodescriptor())
    assert got[: len(expected)] == expected


def test_backlink_protodescriptor_ordering():
    expected = [
        "name",
        "owner",
        "ordinal",
        "annotation",
        "hint",
        "type",
        "nullable",
        "classvar",
        "load",
        "unique",
        "via",
        "limit",
        "repr",
        "doc",
        "config",
    ]
    got = _keys(BackLinkProtodescriptor())
    assert got[: len(expected)] == expected
