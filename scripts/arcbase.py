from datetime import datetime
from typing import Optional

import anaximander as nx
import pandas as pd
from pydantic import BaseModel, Field


@nx.archetype
class MyObject(nx.Object.Base):
    x: bool = nx.metadata(True, typespec=True)
    y: bool = nx.metadata(typespec=True)


@nx.archetype
class DerivedObject(MyObject.Base):
    y = True


@nx.trait("true")
class TrueObject(MyObject):
    @property
    def true(self):
        return self.x


@nx.trait("maybe")
class MaybeObject(DerivedObject):
    @property
    def maybe(self):
        return self.y
