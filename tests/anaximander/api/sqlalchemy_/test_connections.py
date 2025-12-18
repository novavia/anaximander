"""Tests for PostgreSQL connection utilities: psycopg2 connection string generation
and SQLAlchemy engine creation/conversion."""
import psycopg2
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from anaximander.api.sqlalchemy_ import connections
from anaximander.utils import KwargMap


def test_psycopg2_connection_string(postgresql):
    """Verify psycopg2_connection_string produces a usable connection string.

    Args:
        postgresql: Fixture providing connection parameters via .info.

    Asserts:
        A simple SELECT 1 succeeds using a psycopg2 connection built from the string.
    """
    params = KwargMap(postgresql.info, _=["host", "port", "dbname", "user", "password"])
    conn_string = connections.psycopg2_connection_string(**params)
    with psycopg2.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            assert cur.fetchone() == (1,)


def test_postgresql_engine(postgresql):
    """Validate postgresql_engine connectivity and round-trip to psycopg2 string.

    Args:
        postgresql: Fixture providing connection parameters via .info.

    Asserts:
        - Engine connects and SELECT 1 returns a truthy scalar.
        - Converting the Engine back to a psycopg2 string matches the direct one.
    """
    params = KwargMap(postgresql.info, _=["host", "port", "dbname", "user", "password"])
    engine: Engine = connections.postgresql_engine(**params)
    assert engine.connect()
    with Session(engine) as session:
        assert session.scalar(select(1))
    conn_string = connections.psycopg2_connection_string(**params)
    assert connections.postgresql_engine_to_psycopg2_connection_string(engine) == conn_string
