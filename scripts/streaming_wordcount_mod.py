#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""A streaming word-counting workflow.
"""

# pytype: skip-file

import argparse
import logging

import apache_beam as beam
from apache_beam.examples.wordcount_with_metrics import WordExtractingDoFn
from apache_beam.options.pipeline_options import (PipelineOptions,
                                                  SetupOptions,
                                                  StandardOptions)
from apache_beam.transforms import window
from google.cloud import pubsub_v1

PROJECT_ID = "anaximander-tests"

INPUT_TOPIC_ID = "word-in"
OUTPUT_TOPIC_ID = "word-out"

PUB_CLIENT = pubsub_v1.PublisherClient()
SUB_CLIENT = pubsub_v1.SubscriberClient()
INPUT_TOPIC_PATH = PUB_CLIENT.topic_path(PROJECT_ID, INPUT_TOPIC_ID)
OUTPUT_TOPIC_PATH = PUB_CLIENT.topic_path(PROJECT_ID, OUTPUT_TOPIC_ID)


def run(argv=None, save_main_session=True):
  """Build and run the pipeline."""
  parser = argparse.ArgumentParser()
  # group = parser.add_mutually_exclusive_group(required=True)
  known_args, pipeline_args = parser.parse_known_args(argv)

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
  pipeline_options.view_as(StandardOptions).streaming = True
  with beam.Pipeline(options=pipeline_options) as p:

    # Read from PubSub into a PCollection.
    messages = (
        p
        | beam.io.ReadFromPubSub(
            topic=INPUT_TOPIC_PATH).with_output_types(bytes))

    lines = messages | 'decode' >> beam.Map(lambda x: x.decode('utf-8'))

    # Count the occurrences of each word.
    def count_ones(word_ones):
      (word, ones) = word_ones
      return (word, sum(ones))

    counts = (
        lines
        | 'split' >> (beam.ParDo(WordExtractingDoFn()).with_output_types(str))
        | 'pair_with_one' >> beam.Map(lambda x: (x, 1))
        | beam.WindowInto(window.FixedWindows(15, 0))
        | 'group' >> beam.GroupByKey()
        | 'count' >> beam.Map(count_ones))

    # Format the counts into a PCollection of strings.
    def format_result(word_count):
      (word, count) = word_count
      return '%s: %d' % (word, count)

    output = (
        counts
        | 'format' >> beam.Map(format_result)
        | 'encode' >>
        beam.Map(lambda x: x.encode('utf-8')).with_output_types(bytes))

    # Write to PubSub.
    # pylint: disable=expression-not-assigned
    output | beam.io.WriteToPubSub(OUTPUT_TOPIC_PATH)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
