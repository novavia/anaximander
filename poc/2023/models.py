import typing

from attr._make import _make_method
from jinja2 import Environment, FileSystemLoader

from codegen.class_printer import prototype
from codegen.dataclass import Data, dataclass

J2ENV = Environment(
    loader=FileSystemLoader("templates/"), trim_blocks=True, lstrip_blocks=True
)
INIT = J2ENV.get_template("dataclass_init.py.j2")


@dataclass
class C:
    x: float
    y: float


class D:
    x: float
    y: float


@prototype
class E:
    x: float
    y: float


def init_script(cls) -> str:
    annotations = {}
    for k, v in typing.get_type_hints(cls).items():
        if isinstance(v, type):
            annotations[k] = v.__name__
        else:
            annotations[k] = str(v)
    return INIT.render(annotations=annotations)


def make_method(name, script, filename, globs):
    locs = {}
    code = compile(script, "<string>", mode="exec")
    eval(code, globs, locs)
    return locs[name]



D.__init__ = _make_method("__init__", init_script(D), "<string>", {})


if __name__ == "__main__":
    assert issubclass(C, Data)
    assert C.prototype.__annotations__ == {"x": float, "y": float}
    c = C(0, 1)
    assert c.x == 0 and c.y == 1
    # d = D(0, 1)
    # assert d.x ==0 and d.y == 1
    e = E(0, 1)
    assert e.x == 0 and e.y == 1
    # print(inspect.getsource(C.__init__))
    # print(inspect.getsource(D.__init__))
    # print(script(E))
