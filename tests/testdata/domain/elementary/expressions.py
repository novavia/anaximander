from datetime import datetime

import anaximander as nx

DT_FORMAT = "%Y-%m-%d %H:%M:%S"


@nx.compile("dataclasses")
class MyModel(nx.Model):
    t: datetime = nx.field(default=datetime.strptime("2022-01-01 00:00:00", DT_FORMAT))
