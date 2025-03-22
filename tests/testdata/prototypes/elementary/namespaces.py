import anaximander as nx

X = 1

@nx.compile("dataclasses")
class M(nx.Model):
    x = 2
    a: int = nx.field(default=X)
    b: int = nx.field(default=x)
