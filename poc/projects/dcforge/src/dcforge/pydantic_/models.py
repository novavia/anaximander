from pydantic import BaseModel

class MyModel(BaseModel):
    x: int
    y: list[str]
    z: str | None

class MySubModel(BaseModel):
    x: int
    y: list[str]
    z: int

