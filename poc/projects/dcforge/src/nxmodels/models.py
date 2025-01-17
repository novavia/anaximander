import anaximander as nx


@nx.compile("dataclasses")
@nx.compile("pydantic")
class MyModel(nx.Model):
    x: int = nx.Field()
    y: list[str] = nx.Field()
    z: str | None = nx.Field()


@nx.compile("pydantic")
class MySubModel(MyModel):
    z: int = nx.Field()
