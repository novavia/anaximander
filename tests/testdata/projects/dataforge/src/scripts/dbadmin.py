from dataforge.sqlalchemy_.dbconnect import dev_pgsql_engine
from dataforge.sqlalchemy_.target import Base

engine = dev_pgsql_engine()
assert engine.connect()
Base.metadata.create_all(engine)
