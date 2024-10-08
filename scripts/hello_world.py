from datetime import datetime, timedelta
from re import L
from typing import Optional

import numpy as np
import pandas as pd

import anaximander as nx
from anaximander.operators import Sessionizer

# =========================================================================== #
#                              Model declarations                             #
# =========================================================================== #

# Entities are identifiable things
class Machine(nx.Entity):
    id: int = nx.id()
    machine_type: str = nx.data()
    machine_floor: Optional[str] = nx.data()


# Measurements feature units that get printed
# The metadata is carried into model schemas that use this data type and
# can be used by plotting libraries, or for unit conversions.
# Also note the validation input (greater or equal to -273), using
# Pydantic's notations.
class Temperature(nx.Measurement):
    unit = "Celsius"
    ge = -273



class Heartbeat(nx.Sample):
    machine_id: int = nx.key()
    timestamp: datetime = nx.timestamp(freq="1T")

# Samples are timestamped records expected to show up at a somewhat set frequency,
# though not necessarily strictly so. In other words, the freq metadata is
# used as a time characteristic in summarization operations, but missing or
# irregular samples are tolerated.
# Note that the 'machine_id' field will eventually be replaced by a relational
# 'machine' field of type Machine. This functionality is still pending.
# Also note that the temperature field defines its own validation parameters,
# supplemental to those already defined in the Temperature class (not easy!)
class TemperatureSample(nx.Sample):
    machine_id: int = nx.key()
    timestamp: datetime = nx.timestamp(freq="5T")
    temperature: Temperature = nx.data(ge=0, le=200)



class Connectivity(nx.Transition):
    machine_id: int = nx.key()
    timestamp: datetime = nx.timestamp()
    connectivity: str = nx.state()

    @nx.source
    def from_heartbeats(cls):
        return Heartbeat.group_by_key().session_windows().switch(state="connectivity", start="connected", end="disconnected")


# Unlike samples, Journals are strictly periodic -by construction, since they
# are intended as regular summaries, and hence feature a period field, whose
# type is a pandas Period.
class TemperatureJournal(nx.Journal):
    machine_id: int = nx.key()
    period: pd.Period = nx.period(freq="1H")
    avg_temp: Temperature = nx.data()
    min_temp: Temperature = nx.data()
    max_temp: Temperature = nx.data()
    reporting: nx.Percentage = nx.data()
    
    @nx.source
    def from_samples(cls):
        
        def summarizer(cls, machine_id: int, period: datetime, samples: nx.Log[TemperatureSample]):
            temperatures = samples["temperature"]
            avg_temp = temperatures.mean()
            min_temp = temperatures.min()
            max_temp = temperatures.max()
            reporting = 100 * len(samples) / (cls.period  / TemperatureSample.freq)
            return (machine_id, period, avg_temp, min_temp, max_temp, reporting)
        
        return TemperatureSample.group_by_key() \
            .fixed_windows(period=cls.period) \
            .summarize(summarizer) \


# Spec models are intended as general-purpose nested documents, for storing
# specifications, configuration, etc. They have no identifier because they
# always have an 'owner' -typically an Entity or Record object. This bit is
# not implemented yet. If it was, the Machine model would carry an operating
# spec as a data attribute.
# Here the spec defines the nominal operating temperature range, and will
# be used to compute overheat sessions.
class MachineOperatingSpec(nx.Spec):
    min_temp: Temperature = nx.data()
    max_temp: Temperature = nx.data()


# Sessions are timestamped-records with two entries: a start and end times.
# These are ubiquitous in natural data processing, particularly for aggregating
# events, such as oveheat events in this case.

class OverheatSession(nx.Session):
    machine_id: int = nx.key()
    start_time: datetime = nx.start_time()
    end_time: datetime = nx.end_time()
    
    @nx.source
    def from_samples(cls):
        
        def summarizer(cls, machine_id: int, start_time: datetime, end_time: datetime, samples: nx.Log[TemperatureSample]):
            return (machine_id, start_time, end_time)
        
        return TemperatureSample.group_by_key().session_windows(timeout="20T").summarize(summarizer)
    

class Overheating(nx.Transition):
    machine_id: int = nx.key()
    timestamp: datetime = nx.timestamp()
    status: str = nx.state()
    
    @nx.source
    def from_overheat_sessions(cls):
        def merge_map(status, connectivity):
            if connectivity == "disconnected":
                return "disconnected"
            return status
        
        return OverheatSession \
            .filter(lambda s: s.duration > timedelta(minutes="20T")) \
            .switch("status", start="overheat", end="nominal") \
            .merge(Connectivity, merge_map=merge_map)