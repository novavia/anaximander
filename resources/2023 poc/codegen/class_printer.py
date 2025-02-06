import importlib.util
import sys
import types
import typing
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

CODEGEN = Path("__codegen__")
J2ENV = Environment(
    loader=FileSystemLoader("templates/"), trim_blocks=True, lstrip_blocks=True
)
DATACLASS = J2ENV.get_template("dataclass.py.j2")


def script(cls: type) -> types.CodeType:
    annotations = {}
    for k, v in typing.get_type_hints(cls).items():
        if isinstance(v, type):
            annotations[k] = v.__name__
        else:
            annotations[k] = str(v)
    return DATACLASS.render(name=cls.__name__, annotations=annotations)


def print_module(path: Path, contents: str):
    with open(path, "w") as f:
        f.write(contents)


def prototype(cls: type) -> type:
    caller_module = sys.modules[cls.__module__]
    caller_path = Path(caller_module.__file__)
    repo_path = caller_path.parent / "my_app/dataclasses"
    class_name = cls.__name__
    module_name = class_name.lower()
    path = (repo_path / module_name).with_suffix(".py")
    contents = script(cls)
    print_module(path, contents)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return_cls = getattr(module, class_name)
    return_cls.__prototype__ = cls
    return return_cls
