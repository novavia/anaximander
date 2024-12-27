import attrs


@attrs.define
class metadescriptor:
    name: str = attrs.field(init=False)

    def __set_name__(self, owner: type, name: str):
        self.name = name


@attrs.define
class field(metadescriptor):
    name: str = attrs.field(init=False)
    annotation: str | None = attrs.field(init=False)

    def __set_name__(self, owner: type, name: str):
        super().__set_name__(owner, name)
        self.annotation = getattr(owner, "__annotations__", {}).get(name)

    @property
    def hint(self):
        annotation = self.annotation
        if isinstance(annotation, type):
            return annotation.__name__
        return annotation
