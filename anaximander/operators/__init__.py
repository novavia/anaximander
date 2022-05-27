from datetime import timedelta
from typing import Optional, TypeVar, Generic, Any

import pandas as pd

from ..dataobjects import DataObject, Table

IN = TypeVar("IN", bound=DataObject)
OUT = TypeVar("OUT", bound=DataObject)


class Operator(Generic[IN, OUT]):
    def __init__(self, input_type: type[IN], output_type: type[OUT]):
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, input_object: IN) -> OUT:
        pass


class Sessionizer(Operator):
    def __init__(
        self,
        input_type: type[IN],
        output_type: type[OUT],
        feature: str,
        timedelta: Optional[pd.Timedelta] = None,
    ):
        super().__init__(input_type, output_type)
        self.feature = feature
        self.timestamp = self.input_type.dataschema.index.nxtime
        freq = self.input_type.dataschema.fields[self.timestamp].extensions.get("freq")
        self.timedelta = timedelta or pd.Timedelta(freq)
        self.identifier = self.input_type.dataschema.index.nxkey

    def __call__(self, input_object: Table, key: Any, threshold: float) -> OUT:
        data = input_object.data
        feature = self.feature
        timedelta = self.timedelta
        timestamp = self.timestamp
        identifier = self.identifier
        over = data[data[feature] > threshold].reset_index()
        over_index = over[timestamp].to_frame().set_index(over[timestamp])
        gaps = over_index[timestamp].diff() > timedelta
        clusters = gaps.cumsum()
        groups = clusters.groupby(clusters).groups
        session_data = []
        for group in groups.values():
            first, last = group[0], group[-1]
            # avg = data.loc[key][first:last].mean()
            start_time, end_time = first - timedelta / 2, last + timedelta / 2
            session = {
                identifier: key,
                "start_time": start_time,
                "end_time": end_time,
            }
            session_data.append(session)
        session_data = pd.DataFrame.from_records(session_data)
        return Table[self.output_type](session_data)
