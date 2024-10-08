import bytewax.operators as op
from bytewax.connectors.stdio import StdOutSink
from bytewax.dataflow import Dataflow
from bytewax.testing import TestingSource

flow = Dataflow("a_simple_example")

stream = op.input("input", flow, TestingSource(range(10)))


def times_two(inp: int) -> int:
    return inp * 2


double = op.map("double", stream, times_two)

op.output("out", double, StdOutSink())