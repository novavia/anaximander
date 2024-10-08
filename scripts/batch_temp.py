
import argparse
import json
import logging
import random
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib import request

import apache_beam as beam
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.transforms.window import FixedWindows, TimestampedValue
from google.api_core.exceptions import NotFound
from google.cloud import pubsub_v1
from sessionize import sessionize

import anaximander as nx

TEMPFLOW_CSV = nx.TESTDATA / "POC Data - 2023-12-18.csv"
TEMPFLOW_DF = pd.read_csv(TEMPFLOW_CSV)
TEMPFLOW_DF["process_time"] = pd.to_datetime(TEMPFLOW_DF["process_time"])
TEMPFLOW_DF = TEMPFLOW_DF.set_index("process_time").sort_index()
TEMPFLOW = TEMPFLOW_DF.to_dict(orient="index")


class Output(beam.PTransform):
    class _OutputFn(beam.DoFn):

        def process(self, element):
            print(element)

    def expand(self, input):
        input | beam.ParDo(self._OutputFn())


class AddTimestamp(beam.DoFn):
    def process(self, element, **kwargs):
        event_time = element.event_time
        timestamp = datetime.strptime(event_time, "%Y-%m-%d %H:%M:%S").timestamp()
        yield TimestampedValue(element, timestamp)


def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        input = p | beam.io.ReadFromCsv(str(TEMPFLOW_CSV)) | 'timestamp' >> beam.ParDo(AddTimestamp()) # | beam.WithKeys(lambda e: e.machine_id)
        output = input | 'window' >> beam.WindowInto(FixedWindows(60)) \
            | beam.combiners.Count.Globally().without_defaults() \
            | beam.LogElements(with_window=True)
        # output = input | beam.WindowInto(FixedWindows(60)) | Output()
        # (p | beam.Create(['Hello Beam'])
        # | Output())
    # with beam.Pipeline() as p:
    #     # Read the text file[pattern] into a PCollection.
    #     lines = p | 'Read' >> ReadFromText(known_args.input) \
    #             | beam.Filter(lambda line: line != "")

    #     # Write the output using a "Write" transform that has side effects.
    #     # pylint: disable=expression-not-assigned
    #     output = lines | 'Write' >> WriteToText(known_args.output)


    # result = p.run()
    # result.wait_until_finish()
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()