import pandas as pd


def sessionize(event_times, timeout='0s', buffer='0s', onset_span=None):
    timeout = pd.Timedelta(timeout)
    buffer = pd.Timedelta(buffer)
    # Extract the timestamp series of the events
    timestamps = pd.Series(event_times, index=event_times)
    # Compute consecutive differences and mark gaps
    gaps = timestamps.diff() > timeout
    # The cumulative sum of gaps provide clusters of events
    clusters = gaps.cumsum()
    groups = clusters.groupby(clusters).groups
    # Session spans provided by first and last index of each group
    spans = []
    for g in groups.values():
        lower = g[0]
        upper = g[-1]
        spans.append((lower, upper))
    # Check for onset session, and merge it if necessary
    if onset_span is not None:
        onset_span = tuple(onset_span)
        lower = onset_span[1]
        spans = [s for s in spans if s[0] > lower]
        spans.append(onset_span)
    # Extract qualified spans based on buffer
    qspans = [s for s in spans if (s[1] - s[0]) >= buffer]
    return qspans