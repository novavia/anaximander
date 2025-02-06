import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from anaximander import REPO

TESTDATA = REPO / "tests/testdata"
TEMPDATA = REPO / "tests/tempdata"


@contextmanager
def fixturize(path: Path) -> Generator[Path, None, None]:
    """Creates a temporary copy of a file or directory into tempdata for use in tests."""
    if not path.is_relative_to(TESTDATA):
        msg = "Only files and directories in the testdata folder can be fixtureized."
        raise ValueError(msg)
    relative_path = path.relative_to(TESTDATA)
    destination = TEMPDATA / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        path.copy(destination)
    else:
        shutil.copytree(path, destination)
    yield destination
    try:
        shutil.rmtree(destination)
    except FileNotFoundError:
        pass
