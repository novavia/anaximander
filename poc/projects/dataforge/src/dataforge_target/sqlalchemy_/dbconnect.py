import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def pgsql_connection_string(**kwargs: str) -> str:
    """Generates the psycopg2 connection string from keyword arguments."""
    db_host = kwargs.pop("db_host", "localhost")
    db_port = kwargs.pop("db_port", 5432)
    db_name = kwargs.pop("db_name", "postgres")
    db_user = kwargs.pop("db_user", "postgres")
    db_password = kwargs.pop("db_password", "")
    conn_string = f"host={db_host} port={db_port} dbname={db_name} user='{db_user}'"
    if db_password:
        conn_string += f" password={db_password}"
    if "sslmode" in kwargs:
        conn_string += f" sslmode={kwargs.pop('sslmode')}"
    return conn_string


def pgsql_engine(**kwargs: str) -> Engine:
    """Returns a SQLAlchemy Engine object from keyword arguments."""
    db_host = kwargs.pop("db_host", "localhost")
    db_port = kwargs.pop("db_port", 5432)
    db_name = kwargs.pop("db_name", "postgres")
    db_user = kwargs.pop("db_user", "postgres")
    db_password = kwargs.pop("db_password", "")
    conn_string = f"{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    if "sslmode" in kwargs:
        conn_string += f"?sslmode={kwargs.pop('sslmode')}"
    kwargs.setdefault("pool_pre_ping", True)
    kwargs.setdefault("pool_recycle", 600)
    engine = create_engine("postgresql+psycopg2://" + conn_string, **kwargs)
    return engine


def dev_pgsql_connection_string(**kwargs) -> str:
    """Generates the psycopg2 connection string for the local dev database.

    This assumes that a local database has been created in the development
    environment, and that the connection parameters are stored in a .env
    file -or otherwise exist as environment variables.
    """
    db_host = os.environ["DEV_DB_HOST"]
    db_name = os.environ["DEV_DB_NAME"]
    db_user = os.environ["DEV_DB_USER"]
    db_password = os.environ["DEV_DB_PASSWORD"]
    return pgsql_connection_string(db_host=db_host, db_name=db_name,
                                   db_user=db_user, db_password=db_password)


def dev_pgsql_engine(**kwargs) -> Engine:
    """Returns a SQLAlchemy Engine object for the dev database."""
    db_host = os.environ["DEV_DB_HOST"]
    db_name = os.environ["DEV_DB_NAME"]
    db_user = os.environ["DEV_DB_USER"]
    db_password = os.environ["DEV_DB_PASSWORD"]
    return pgsql_engine(db_host=db_host, db_name=db_name,
                        db_user=db_user, db_password=db_password, **kwargs)