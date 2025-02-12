from typing import TypedDict

from sqlalchemy import JSON
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    type_annotation_map = {
        TypedDict: JSON,
        dict: JSON,
        list: JSON,
        tuple: JSON,
        set: JSON,
    }
