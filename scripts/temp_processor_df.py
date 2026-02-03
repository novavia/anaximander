
import argparse
import json
import logging
import random
import socket
import typing
from collections import Counter
from concurrent.futures import TimeoutError
from datetime import datetime

import pandas as pd
from apache_beam import (DoFn, Filter, GroupByKey, Map, ParDo, Pipeline,
                         PTransform, Select, WindowInto, WithKeys, io)
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.trigger import (AccumulationMode, AfterAny,
                                            AfterCount, AfterProcessingTime,
                                            AfterWatermark, Repeatedly)
from apache_beam.transforms.window import FixedWindows, Sessions
from apache_beam.utils.timestamp import Duration
from google.cloud import pubsub_v1
from pexpect import TIMEOUT
from sessionize import sessionize

import anaximander as nx

PROJECT_ID = "anaximander-tests"
SUBSCRIPTION_ID = "tempflow-sub"
SUBSCRIPTION_PATH = f"projects/{PROJECT_ID}/subscriptions/{SUBSCRIPTION_ID}"

class TempMessage(typing.NamedTuple):
    machine_id: int
    event_time: str
    temperature: float


class GroupMessagesByFixedWindows(PTransform):
    """A composite transform that groups Pub/Sub messages based on publish time
    and outputs a list of tuples, each containing a message and its publish time.
    """

    def __init__(self, window_size, num_shards=1):
        # Set window size to 30 seconds.
        self.window_size = int(window_size * 30)
        self.num_shards = num_shards

    def expand(self, pcoll):
        return (
            pcoll
            # | Map(lambda e: json.loads(e)).with_output_types(TempMessage)
            | Map(lambda e: json.loads(e))
            | Filter(lambda e: e["temperature"] >= 55)
            # Bind window info to each element using element timestamp (or publish time).
            | "Window into fixed intervals"
            >> WindowInto(Sessions(self.window_size), accumulation_mode=AccumulationMode.ACCUMULATING, allowed_lateness=60*24*3600)
            # | "Window into fixed intervals"
            # >> WindowInto(SlidingWindows(self.window_size, self.window_size), allowed_lateness=0)
            | "Add timestamp to windowed elements" >> ParDo(AddTimestamp())
            # Assign a random key to each windowed element based on the number of shards.
            | "Add key" >> WithKeys(lambda _: random.randint(0, self.num_shards - 1))
            # Group windowed elements by key. All the elements in the same window must fit
            # memory for this. If not, you need to use `beam.util.BatchElements`.
            | "Group by key" >> GroupByKey()
        )

class AddTimestamp(DoFn):
    def process(self, element, publish_time=DoFn.TimestampParam):
        """Processes each windowed element by extracting the message body and its
        publish time into a tuple.
        """
        timestamp = datetime.utcfromtimestamp(float(publish_time)).strftime("%Y-%m-%d %H:%M:%S.%f")
        # timestamp = element["event_time"]
        print(timestamp)
        # yield (element.decode("utf-8"), timestamp)
        yield (element, timestamp)


class WriteToGCS(DoFn):
    def __init__(self, output_path):
        self.output_path = output_path

    def process(self, key_value, window=DoFn.WindowParam):
        """Write messages in a batch to Google Cloud Storage."""

        ts_format = "%H:%M:%S"
        window_start = window.start.to_utc_datetime().strftime(ts_format)
        window_end = window.end.to_utc_datetime().strftime(ts_format)
        window = (window_start, window_end)
        print(window)
        shard_id, batch = key_value
        filename = "-".join([self.output_path, window_start, window_end, str(shard_id)])

        with io.gcsio.GcsIO().open(filename=filename, mode="w") as f:
            for message_body, publish_time in batch:
                f.write(f"{message_body},{publish_time}\n".encode("utf-8"))

def run(window_size=1.0, num_shards=1, pipeline_args=None): # Set `save_main_session` to True so DoFns can access globally imported modules.
    gcs_path = "gs://dfdump/f"
    # local_path = "~/anaximander/tests/data/dfdump"
    pipeline_options = PipelineOptions(
    pipeline_args, streaming=True, save_main_session=True, allow_unsafe_triggers=True
    )

    with Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            # Because `timestamp_attribute` is unspecified in `ReadFromPubSub`, Beam
            # binds the publish time returned by the Pub/Sub server for each message
            # to the element's timestamp parameter, accessible via `DoFn.TimestampParam`.
            # https://beam.apache.org/releases/pydoc/current/apache_beam.io.gcp.pubsub.html#apache_beam.io.gcp.pubsub.ReadFromPubSub
            | "Read from Pub/Sub" >> io.ReadFromPubSub(subscription=SUBSCRIPTION_PATH, timestamp_attribute="event_time")
            | "Window into" >> GroupMessagesByFixedWindows(window_size, num_shards)
            | "Write to GCS" >> ParDo(WriteToGCS(gcs_path))
            # | "Write to file" >> io.WriteToText(local_path)
        )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window_size",
        type=float,
        default=1.0,
        help="Output file's window size in minutes.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Number of shards to use when writing windowed elements to GCS.",
    )
    known_args, pipeline_args = parser.parse_known_args()

    run(
        known_args.window_size,
        known_args.num_shards,
        pipeline_args,
    )
