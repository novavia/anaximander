import types
import typing

from jinja2 import Environment, FileSystemLoader

J2ENV = Environment(loader=FileSystemLoader("templates/"))
INIT = J2ENV.get_template("dataclass_init.py.j2")


class DataType(type):
    """The metaclass for data classes."""

    def __new__(mcl, name, bases, namespace, prototype: type = object):
        synth_ns = {}  # Initializes a synthetic namespace
        exec(mcl.init(prototype), synth_ns)
        synth_ns |= namespace
        return super().__new__(mcl, name, bases, synth_ns)

    def __init__(cls, name, bases, namespace, prototype: type = object):
        cls.__prototype__ = prototype

    @property
    def prototype(cls) -> type:
        return cls.__prototype__

    @classmethod
    def init(mcl, prototype: type) -> types.CodeType:
        annotations = {}
        for k, v in typing.get_type_hints(prototype).items():
            if isinstance(v, type):
                annotations[k] = v.__name__
            else:
                annotations[k] = str(v)
        return compile(INIT.render(annotations=annotations), "<string>", mode='exec')


class Data(metaclass=DataType):
    """Base class for data classes."""
    pass


def dataclass(prototype: type) -> type[Data]:
    """A class decorator that compiles a Data subclass."""
    kwds = dict(prototype=prototype)
    cls = types.new_class(prototype.__name__, bases=(Data,), kwds=kwds, exec_body=None)
    return cls
