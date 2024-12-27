import anaximander as nx


@nx.compile("dataclasses")
class MyModel(nx.Model):
    x: int = nx.field()
    y: str = nx.field()


class MySubModel(MyModel):
    z: int = nx.field()