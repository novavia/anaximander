
import json
from concurrent import futures
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib import request

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import pubsub_v1
from sessionize import sessionize

import anaximander as nx

PROJECT_ID = "anaximander-tests"
TOPIC_ID = "tempflow"
SUBSCRIPTION_ID = "tempflow-sub"

PUB_CLIENT = pubsub_v1.PublisherClient()
SUB_CLIENT = pubsub_v1.SubscriberClient()
TOPIC_PATH = PUB_CLIENT.topic_path(PROJECT_ID, TOPIC_ID)
SUBSCRIPTION_PATH = SUB_CLIENT.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

# Create the topic if it doesn't already exist
try:
    PUB_CLIENT.get_topic(request={"topic": TOPIC_PATH})
except NotFound:
    PUB_CLIENT.create_topic(request={"name": TOPIC_PATH})

try:
    SUB_CLIENT.get_subscription(request={"subscription": SUBSCRIPTION_PATH})
except NotFound:
    SUB_CLIENT.create_subscription(request={"name": SUBSCRIPTION_PATH, "topic": TOPIC_PATH})


# Data preparation
TEMPFLOW_CSV = nx.TESTDATA / "POC Data - 2023-12-18.csv"
TEMPFLOW_DF = pd.read_csv(TEMPFLOW_CSV)
TEMPFLOW_DF["process_time"] = pd.to_datetime(TEMPFLOW_DF["process_time"])
TEMPFLOW_DF = TEMPFLOW_DF.set_index("process_time").sort_index()
TEMPFLOW = TEMPFLOW_DF.to_dict(orient="index")


# Message publication
publish_futures = []

def get_callback(
    publish_future: pubsub_v1.publisher.futures.Future, data: str
) -> Callable[[pubsub_v1.publisher.futures.Future], None]:
    def callback(publish_future: pubsub_v1.publisher.futures.Future) -> None:
        try:
            # Wait 60 seconds for the publish call to succeed.
            print(publish_future.result(timeout=60))
        except futures.TimeoutError:
            print(f"Publishing {data} timed out.")

    return callback

for record in TEMPFLOW.values():
    data = json.dumps(record)
    # When you publish a message, the client returns a future.
    publish_future = PUB_CLIENT.publish(TOPIC_PATH, data.encode("utf-8"), event_time=record["event_time"])
    # Non-blocking. Publish failures are handled in the callback function.
    publish_future.add_done_callback(get_callback(publish_future, data))
    publish_futures.append(publish_future)

# Wait for all the publish futures to resolve before exiting.
futures.wait(publish_futures, return_when=futures.ALL_COMPLETED)

print(f"Published messages with error handler to {TOPIC_PATH}.")