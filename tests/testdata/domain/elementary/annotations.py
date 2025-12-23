import anaximander as nx

@nx.compile("dataclasses")
class A(nx.Model):
    a: int = nx.field()
    b: "B" = nx.field()


@nx.compile("dataclasses")
class B(nx.Model):
    a: A = nx.field()
    b: str = nx.field()
