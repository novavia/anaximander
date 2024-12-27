import types
import typing



class Data:
    """Base class for data classes."""
    pass


# def dataclass(prototype: type) -> type[Data]:
#     """A class decorator that compiles a Data subclass."""
#     kwds = dict(prototype=prototype)
#     cls = types.new_class(prototype.__name__, bases=(Data,), kwds=kwds, exec_body=None)
#     return cls
