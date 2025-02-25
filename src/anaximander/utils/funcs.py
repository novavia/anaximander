import contextlib
import itertools
import os
import re
import socket
from pathlib import Path
from types import ModuleType
from typing import Generator


import inflect
from inflect import Word

IE = inflect.engine()


def boolean(string):
    """Converts a string to a boolean."""
    if string == "True":
        return True
    elif string == "False":
        return False
    else:
        raise ValueError


def is_online():
    """Function that determines if the tester is online."""
    connection = None
    try:
        host = socket.gethostbyname("www.google.com")
        connection = socket.create_connection((host, 80), 2)
    except:  # noqa
        return False
    else:
        return True
    finally:
        if connection is not None:
            connection.close()


def offline(assertion=None):
    """Returns application status, or sets it if assertion is passed.

    Asserting offline is False is subject to verifying that the application
    truly is online.
    """
    if assertion is not None:
        if assertion is False:
            try:
                assert is_online()
            except AssertionError:
                pass
            else:
                os.environ["OFFLINE"] = "False"
                return False
        os.environ["OFFLINE"] = "True"
        return True
    try:
        return boolean(os.environ["OFFLINE"])
    except KeyError:
        which = not is_online()
        os.environ["OFFLINE"] = str(which)
        return which


def local_runtime() -> bool:
    """Returns True if an app runtime executes locally, False in the cloud environment."""
    return bool(os.getenv("IS_RUNNING_LOCALLY"))


def is_close(a, b, tolerance=1e-9):
    """Near-equality test function."""
    return abs(a - b) < tolerance


def batched(iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def subclasses(cls, depth: int = -1, strict: bool = True) -> list[type]:
    """Recursively get all subclasses of a class, to a given inheritance depth.

    Args:
        depth (int, optional): Sets the inheritance depth. Defaults to -1,
            which means all subclasses.
        strict (bool, optional): If True, cls itself is excluded from the results.

    Returns:
        list[type]: An list of subclasses.
    """
    all = []
    if not strict:
        all.append(cls)
    if depth != 0:
        subs = cls.__subclasses__()
        all.extend(subs)
        for sub in subs:
            all.extend(subclasses(sub, depth=depth - 1))
    return all


def camel_to_snake(s: str) -> str:
    """Converts a camel case string to a snake case string.


    Args:
        s (str): string to convert.

    Returns:
        str: converted string.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    s2 = re.sub("__([A-Z])", r"_\1", s1)
    s3 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s2)
    return s3.lower()


def pluralize(s: str) -> str:
    """Pluralizes a string.

    Args:
        s (str): string to pluralize.

    Returns:
        str: pluralized string.
    """
    if isinstance(s, Word):
        if not IE.singular_noun(s):
            return IE.plural(s)
    return s


def type_name_to_collection_name(name: str) -> str:
    """Converts a class name to a collection name.

    Effectively converts CamelCase to snake_case and adds plural form.

    Args:
        name (str): A class name presumed to be CamelCase.


    Returns:
        str: A collection name using snake_case and plural form.
    """
    snake = camel_to_snake(name)
    *prefixes, last_word = snake.split("_")
    if isinstance(last_word, Word):
        last_word = pluralize(last_word)
    return "_".join(prefixes + [last_word])


@contextlib.contextmanager
def workdir(path: Path | str, *, mkdir: bool = True) -> Generator[Path, None, None]:
    """A context manager that temporarily changes the working directory, optionally setting it."""
    path = Path(path)
    cwd = Path.cwd()
    if mkdir:
        if not path.exists():
            path.mkdir(parents=True)
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(cwd)


def is_package_init(module: ModuleType):
    if not (file_path := getattr(module, '__file__', None)):
        return False
    path = Path(file_path)
    return path.stem == "__init__"
