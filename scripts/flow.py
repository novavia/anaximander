
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bytewax.operators as op
import bytewax.operators.window as win
import pandas as pd
from bytewax.connectors.stdio import StdOutSink
from bytewax.dataflow import Dataflow
from bytewax.operators.window import EventClockConfig, SessionWindow
from bytewax.testing import TestingSource
from sessionize import sessionize

import anaximander as nx

flow = Dataflow("sessionize")


TEMPFLOW_CSV = nx.TESTDATA / "POC Data - 2023-12-18.csv"


TEMPFLOW_DF = pd.read_csv(TEMPFLOW_CSV)
DT_COLS = ["Event Time", "Process Time"]
for col in DT_COLS:
    TEMPFLOW_DF[col] = pd.to_datetime(TEMPFLOW_DF[col])
TEMPFLOW_DF = TEMPFLOW_DF.set_index("Process Time").sort_index()

TEMPFLOW = TEMPFLOW_DF.to_dict(orient="index")
THRESHOLD = 55

HIGH_TEMPS_DF = TEMPFLOW_DF[TEMPFLOW_DF["Temperature"] >= THRESHOLD]
HIGH_TEMPS = HIGH_TEMPS_DF["Event Time"]

SESSIONS = sessionize(HIGH_TEMPS.values, timeout="30s", buffer="30s")


@dataclass
class TempEvent:
    machine_id: int
    event_time: datetime
    temperature: int
    
    def __post_init__(self):
        self.event_time = pd.to_datetime(self.event_time).to_pydatetime().astimezone(timezone.utc)
    

TEMP_EVENTS = [TempEvent(*r) for r in HIGH_TEMPS_DF.values]

def shuffle(events: list):
    temp = events.copy()
    temp[0], temp[1] = temp[1], temp[0]
    return temp

events = op.input("input", flow, TestingSource(shuffle(TEMP_EVENTS)))

def machine_event(event: TempEvent):
    return str(event.machine_id), event


keyed_events = op.map("key on machines", events, machine_event)


def add_event(acc, event):
    acc.append(event)
    return acc


clock_config = EventClockConfig(
    lambda e: e.event_time, wait_for_system_duration=timedelta(seconds=60)
)
window_config = SessionWindow(gap=timedelta(seconds=30))
sessions = win.fold_window("session_windows", keyed_events, clock_config, window_config, list, add_event)
# ('1', [Search(user=1, query='dogs', time=datetime.datetime...)])


def head_and_tail(machine_overheat_session):
    machine, (session, events) = machine_overheat_session
    event_times = [e.event_time for e in events]
    head, tail = min(event_times), max(event_times)
    return [(machine, head, "hot"), (machine, tail, "nominal")]
    


transition_pairs = op.map("transition_pairs", sessions, head_and_tail)
transitions = op.flatten("transitions", transition_pairs)
# ('1', 1.0)
# ('2', 2.0)
op.output("stdout", transitions, StdOutSink())