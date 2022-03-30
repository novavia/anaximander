from anaximander.descriptors import data
from anaximander.descriptors.base import yaml
from anaximander.descriptors.schema import DataSchema
from anaximander.meta import *
from anaximander.dataobjects.base import Spec
from anaximander.dataobjects.datafields import DataField


class Point(Spec):
    x: int = data()
    y: int = data()


schema = Point.schema
x = schema.fields["x"]
print(x)
print(schema)
