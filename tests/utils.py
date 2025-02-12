import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from anaximander import REPO
from anaximander.api.sqlalchemy_ import connections
from anaximander.utils import KwargMap

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


@pytest.fixture
def engine(postgresql) -> Generator[Engine, None, None]:
    """A SQLAlchemy engine fixture."""
    params = KwargMap(postgresql.info, _=["host", "port", "dbname", "user", "password"])
    engine: Engine = connections.postgresql_engine(**params)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Generator[Session, None, None]:
    """A SQLAlchemy session fixture."""
    with Session(engine) as session:
        yield session
    session.close()
