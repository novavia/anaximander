from datetime import datetime
from typing import Optional

import anaximander as nx
from anaximander.operators import Sessionizer
import numpy as np
import pandas as pd


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


# Unlike samples, Journals are strictly periodic -by construction, since they
# are intended as regular summaries, and hence feature a period field, whose
# type is a pandas Period.
class TemperatureJournal(nx.Journal):
    machine_id: int = nx.key()
    period: pd.Period = nx.period(freq="1H")
    avg_temp: Temperature = nx.data()
    min_temp: Temperature = nx.data()
    max_temp: Temperature = nx.data()


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


# This is a very rudimentary implementation of a parametric operator that will
# be used to compute overheat sessions. Note that the Sessionizer class reads
# metadata from the TemperatureSample class, such as the names of the key
# and timestamp field, as well as the timestamp frequency.
sessionizer = Sessionizer(TemperatureSample, OverheatSession, feature="temperature")

# In Anaximander, types are composable, and automatically assembled without
# the need for a class declaration. Here we explicitly name a class for
# tables of temperature samples. Unlike the use of generics in type annotations,
# Table[TemperatureSample] is an actual class -which is cached once it is
# created.
# Table is a so-called archetype, and Table[TemperatureSample] is a concrete
# subtype. A table instance wraps a dataframe along with metadata inherited
# from its class -primarily the model's schema, which is used to conform and
# validate the data.
TempTable = nx.Table[TemperatureSample]


# =========================================================================== #
#                                 Data inputs                                 #
# =========================================================================== #

# Machine instance and operating spec
m0 = Machine(id=0, machine_type="motor")
m0_spec = MachineOperatingSpec(min_temp=40.0, max_temp=55.0)

# Building a table of temperature samples
times = pd.date_range(start="2022-2-18 12:00", freq="5T", periods=12)
temperatures = [45.0, 46.0, 45.0, 50.0, 59.0, 50.0, 48.0, 51.0, 52.0, 56.0, 58.0, 53.0]
sample_log = TempTable(dict(machine_id=0, timestamp=times, temperature=temperatures))

# Computing stats over an hour to fill a Journal instance. Note that eventually
# this kind of summarization will be specified in model declarations and
# carried out by operators -which will be at least partially automated, see
# the sessionizer for a prototype.
avg_temp = round(np.mean(temperatures), 0)
min_temp = min(temperatures)
max_temp = max(temperatures)
hourly = TemperatureJournal(
    machine_id=0,
    period="2022-2-18 12:00",
    avg_temp=avg_temp,
    min_temp=min_temp,
    max_temp=max_temp,
)

# Computing overheat sessions
# Note that the key and threshold will not be necessary once relations are
# established between models -the sample log will be able to point to a machine
# as part of its metadata, and the machine will own its operating spec,
# providing a path to the threshold.
overheat_sessions = sessionizer(sample_log, key=0, threshold=m0_spec.max_temp.data)

# =========================================================================== #
#                                 Evaluations                                 #
# =========================================================================== #

# TempTable exposes a data frame, that is automatically indexed by key
# and timestamp
assert isinstance(sample_log.data, pd.DataFrame)
print(sample_log)

# TempTable can broken down into individual records of type TemperatureSample.
# These feature temperature attributes with unit metadata.
# Likewise, the temperature attribute of the TempTable is a series of
# temperatures, and individual data points are measurements with metadata.
r0 = next(sample_log.records())
assert isinstance(r0, TemperatureSample)
print(r0.temperature)
assert next(sample_log.temperature.values()) == r0.temperature

# Temperature defines a lower bound, which is used to validate inputs
try:
    Temperature(-300)
except ValueError as e:
    print(e)

# TemperatureSample defines its own bounds as well
# This would also work if one tried to directly instantiate a table of
# temperature samples from a dataframe -though in the current implementation
# the framework converts the data frame to records and uses Pydantic, which
# is obviously very inefficient. The target is to use the Panderas library,
# with no change in the interface.
try:
    TemperatureSample(machine_id=m0.id, timestamp=times[0], temperature=250)
except ValueError as e:
    print(e)

# Here is a printout of our Journal instance, along with a basic property
print(hourly)

# And finally a printout of our overheating sessions' timespans, a pandas
# IntervalIndex
print(overheat_sessions.timespans)
