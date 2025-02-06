import anaximander as nx


@nx.compile("dataclasses")
class MyModel(nx.Model):
    x: int = nx.Field()
    y: str = nx.Field()


class MySubModel(MyModel):
    z: int = nx.Field()
