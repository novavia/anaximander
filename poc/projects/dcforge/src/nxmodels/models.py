import anaximander as nx


@nx.compile("dataclasses")
@nx.compile("pydantic")
class MyModel(nx.Model):
    x: int = nx.field()
    y: list[str] = nx.field()
    z: str | None = nx.field()

@nx.compile("pydantic")
class MySubModel(MyModel):
    z: int = nx.field()