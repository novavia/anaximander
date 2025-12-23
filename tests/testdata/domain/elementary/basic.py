import anaximander as nx

@nx.compile("dataclasses")
class MyModel(nx.Model):
    a: int = nx.field()
    b: str = nx.field()
