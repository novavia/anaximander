from abc import ABCMeta

from ..metadescriptors import metadescriptor


class Prototype(ABCMeta):
    """Metaclass for model and data declarative types."""

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.__compilations__ = {}

    def metadescriptors(cls, *types: type[metadescriptor], inherited: bool = True):
        """Returns a dictionary of metadescriptors of the supplied types.

        If inherited is set to True, metadescriptors declared in parent models are included.
        Otherwise, only the metadescriptors directly declared by cls are returned.
        """
        if not types:
            types = (metadescriptor,)
        cls_metadescriptors = {k: v for k, v in cls.__dict__.items() if isinstance(v, types)}
        if inherited:
            parent = cls.mro()[1]
            if isinstance(parent, Prototype):
                metadescriptors = (
                    parent.metadescriptors(*types, inherited=True) | cls_metadescriptors
                )
            else:
                metadescriptors = cls_metadescriptors
        else:
            metadescriptors = cls_metadescriptors
        return metadescriptors
