
import json
from concurrent.futures import TimeoutError

import pandas as pd
from google.cloud import pubsub_v1
from pexpect import TIMEOUT
from sessionize import sessionize

import anaximander as nx

PROJECT_ID = "anaximander-tests"
SUBSCRIPTION_ID = "tempflow-sub"

SUB_CLIENT = pubsub_v1.SubscriberClient()
SUBSCRIPTION_PATH = SUB_CLIENT.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
TIMEOUT = 5.0


RECORDS = []

# Message reception
DF = None
def accumulator(record=None):
    records = []
    try:
        while True:
            try:
                record = (yield record)
                records.append(record)
            except Exception as e:
                record = e
    finally:
        global DF
        DF = pd.DataFrame(records)

receiver = accumulator()

# Message processing

def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received {message.data!r}.")
    record = pd.Series(json.loads(message.data.decode("utf-8")))
    # receiver.send(record)
    RECORDS.append(record)
    # ack the received message
    message.ack()

streaming_pull_future = SUB_CLIENT.subscribe(SUBSCRIPTION_PATH, callback=callback)
print(f"Listening for messages on {SUBSCRIPTION_PATH}..\n")

# Wrap subscriber in a 'with' block to automatically call close() when done.
with SUB_CLIENT:
    try:
        # When `timeout` is not set, result() will block indefinitely,
        # unless an exception is encountered first.
        streaming_pull_future.result(timeout=TIMEOUT)
    except TimeoutError:
        streaming_pull_future.cancel()  # Trigger the shutdown.
        streaming_pull_future.result()  # Block until the shutdown is complete.
        receiver.close()
        DF = pd.DataFrame(RECORDS)
        print(DF)