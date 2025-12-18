"""Provides connection strings and SQLAlchemy connection engines."""

import os
from typing import NotRequired, TypedDict, cast

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


class ConnectionParameters(TypedDict):
    host: str
    port: str
    dbname: str
    user: str
    password: NotRequired[str]
    sslmode: NotRequired[str]


def postgresql_connection_parameters(
    host: str | None = None,
    port: str | int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    sslmode: str | None = None,
    *,
    os_environ_prefix: str | None = None,
) -> ConnectionParameters:
    """Casts connection parameters from keyword arguments or environment variables.

    Args:
        host (str | None, optional): the database host. Defaults to None.
        port (str | int | None, optional): the database port. Defaults to None.
        dbname (str | None, optional): the database name. Defaults to None.
        user (str | None, optional): the database user. Defaults to None.
        password (str | None, optional): optional user password. Defaults to None.
        sslmode (str | None, optional): connection ssl mode. Defaults to None.
        os_environ_prefix (str | None, optional): if supplied, looks for parameters in
            environment variables with this prefix.

    Returns:
        ConnectionParameters: a dictionary of connection parameters.
    """
    if os_environ_prefix:
        prefix = os_environ_prefix.upper() + "_"
        environ_host = os.environ.get(prefix + "HOST")
        environ_port = os.environ.get(prefix + "PORT")
        environ_dbname = os.environ.get(prefix + "DBNAME")
        environ_user = os.environ.get(prefix + "USER")
        environ_password = os.environ.get(prefix + "PASSWORD")
        environ_sslmode = os.environ.get(prefix + "SSLMODE")
    else:
        environ_host = None
        environ_port = None
        environ_dbname = None
        environ_user = None
        environ_password = None
        environ_sslmode = None
    params = {
        "host": host or environ_host or "localhost",
        "port": str(port or environ_port or 5432),
        "dbname": dbname or environ_dbname or "postgres",
        "user": user or environ_user or "postgres",
    }
    if pwd := password or environ_password:
        params["password"] = pwd
    if ssl := sslmode or environ_sslmode:
        params["sslmode"] = ssl
    return cast(ConnectionParameters, params)


def psycopg2_connection_string(
    host: str | None = None,
    port: str | int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    sslmode: str | None = None,
    *,
    os_environ_prefix: str | None = None,
    **kwargs,
) -> str:
    """Generates the psycopg2 connection string from keyword arguments.

    Args:
        host (str | None, optional): the database host. Defaults to None.
        port (str | int | None, optional): the database port. Defaults to None.
        dbname (str | None, optional): the database name. Defaults to None.
        user (str | None, optional): the database user. Defaults to None.
        password (str | None, optional): optional user password. Defaults to None.
        sslmode (str | None, optional): connection ssl mode. Defaults to None.
        os_environ_prefix (str | None, optional): if supplied, looks for parameters in
            environment variables with this prefix.

    Returns:
        str: a connection string to supply to psycopg2.
    """
    params = postgresql_connection_parameters(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        os_environ_prefix=os_environ_prefix,
    )
    host = params["host"]
    port = params["port"]
    dbname = params["dbname"]
    user = params["user"]
    password = params.get("password")
    sslmode = params.get("sslmode")
    conn_string = f"host={host} port={port} dbname={dbname} user='{user}'"
    if password:
        conn_string += f" password={password}"
    if sslmode:
        conn_string += f" sslmode={sslmode}"
    return conn_string


def postgresql_engine(
    host: str | None = None,
    port: str | int | None = None,
    dbname: str | None = None,
    user: str | None = None,
    password: str | None = None,
    sslmode: str | None = None,
    *,
    os_environ_prefix: str | None = None,
    **kwargs,
) -> Engine:
    """Return a SQLAlchemy Engine from keyword arguments or environment variables.

    Args:
        host (str | None, optional): The database host. Defaults to None.
        port (str | int | None, optional): The database port. Defaults to None.
        dbname (str | None, optional): The database name. Defaults to None.
        user (str | None, optional): The database user. Defaults to None.
        password (str | None, optional): Optional user password. Defaults to None.
        sslmode (str | None, optional): Connection SSL mode. Defaults to None.
        os_environ_prefix (str | None, optional): If supplied, looks for parameters in
            environment variables with this prefix.

    Returns:
        Engine: A SQLAlchemy Engine object.
    """
    params = postgresql_connection_parameters(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode,
        os_environ_prefix=os_environ_prefix,
    )
    host = params["host"]
    port = params["port"]
    dbname = params["dbname"]
    user = params["user"]
    password = params.get("password")
    sslmode = params.get("sslmode")
    if password:
        conn_string = f"{user}:{password}@{host}:{port}/{dbname}"
    else:
        conn_string = f"{user}@{host}:{port}/{dbname}"
    if sslmode:
        conn_string += f"?sslmode={sslmode}"
    kwargs.setdefault("pool_pre_ping", True)
    kwargs.setdefault("pool_recycle", 600)
    engine = create_engine("postgresql+psycopg2://" + conn_string, **kwargs)
    return engine


def postgresql_engine_to_psycopg2_connection_string(engine: Engine):
    """Convert a SQLAlchemy Engine to a psycopg2 connection string.

    Args:
        engine (Engine): SQLAlchemy Engine configured for a PostgreSQL database.

    Returns:
        str: A psycopg2-style connection string derived from the engine.
    """
    host = engine.url.host
    port = engine.url.port
    dbname = engine.url.database
    user = engine.url.username
    password = engine.url.password
    kwargs = {"host": host, "port": port, "dbname": dbname, "user": user, "password": password}
    ssl_mode = engine.url.query.get("sslmode")
    if ssl_mode:
        kwargs["sslmode"] = ssl_mode
    return psycopg2_connection_string(**kwargs)
