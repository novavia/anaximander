"""This modulde defines the Scope concept, which delineates data extent.

"""

from collections.abc import Collection, Sequence, Mapping
from datetime import date
from typing import ClassVar, Union, Optional, Any, TypedDict

from frozendict import frozendict
import pandas as pd

from .datatypes import nxdata, NoneType


class ScopeMap(TypedDict):
    keys: Optional[
        Union[tuple, frozenset, frozendict[str, Union[tuple, frozenset]]]
    ]  # Either set or set map for compound keys
    time: Optional[pd.Interval]  # These are intervals of pd.Timestamp
    # In latter releases, this may be extended to sets of time intervals
    space: NoneType  # This is a placeholder for pending features


def _freeze(value):
    """A utility that freezes sets and dictionaries recursively."""
    if isinstance(value, Mapping):
        return frozendict({k: _freeze(v) for k, v in value.items()})
    elif isinstance(value, Sequence):
        return tuple(value)
    elif isinstance(value, Collection):
        return frozenset(value)
    else:
        return tuple(
            value,
        )


class DataScope(Mapping):
    """A definition of data bounds in keys, time and space dimensions.

    Data bounds are expressed in terms of singletons, sets or intervals.
    For keys, the entry may be a dictionary in case of a composite key structure.
    """

    _data: ScopeMap

    def __init__(
        self,
        data: Mapping[str, Any],
        keys: Any = None,
        time: Optional[
            Union[date, str, Collection[date], Collection[str], pd.Interval]
        ] = None,
        space=None,
    ):
        if isinstance(data, DataScope) and all(v is None for v in [keys, time, space]):
            self._data = data._data
        else:
            kwargs = dict(keys=keys, time=time, space=space)
            data = data | {k: v for k, v in kwargs.items() if v is not None}
            keys_scope = data.get("keys")
            time_scope = data.get("time")
            space_scope = None
            if keys_scope is not None:
                keys_scope = _freeze(keys_scope)
            if time_scope is not None:
                if isinstance(time_scope, pd.Interval):
                    time_scope = pd.Interval(
                        pd.to_datetime(time_scope.left),
                        pd.to_datetime(time_scope.right),
                        "left",
                    )
                elif isinstance(time_scope, Collection):
                    # Assume this is a pair of timestamps
                    time_scope = pd.Interval(
                        pd.to_datetime(time_scope[0]),
                        pd.to_datetime(time_scope[1]),
                        "left",
                    )
                else:
                    time_scope = pd.Interval(
                        pd.to_datetime(time_scope), pd.to_datetime(time_scope), "both"
                    )

            self._data = frozendict(keys=keys_scope, time=time_scope, space=space_scope)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __eq__(self, other: "DataScope"):
        try:
            return self._data == other._data
        except AttributeError:
            return False

    def __ne__(self, other: "DataScope"):
        try:
            return self._data != other._data
        except AttributeError:
            return False

    def __hash__(self):
        return hash(self._data)

    def __hash__(self):
        return hash(self._data)

    @classmethod
    def __get_validators__(cls):
        # Enables type checking by Pydantic
        yield lambda v: cls(v)
