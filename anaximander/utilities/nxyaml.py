from ruamel.yaml import YAML, yaml_object
from ruamel.yaml.main import add_multi_representer
from ruamel.yaml.representer import BaseRepresenter
from ruamel.yaml.compat import StringIO


original_add_representer = BaseRepresenter.__dict__["add_representer"].__wrapped__
add_multi_representer = BaseRepresenter.__dict__["add_multi_representer"].__wrapped__


def add_representer(cls, data_type, representer):
    original_add_representer(cls, data_type, representer)
    add_multi_representer(cls, data_type, representer)


BaseRepresenter.add_representer = classmethod(add_representer)


class NxYAML(YAML):
    def dumps(self, data, **kwargs):
        stream = StringIO()
        YAML.dump(self, data, stream, **kwargs)
        return stream.getvalue()


yaml = NxYAML()
yaml_representer = yaml.representer

# # def add_representer(cls, data_type, representer):

# #     @classmethod
# #     def add_representer(cls, data_type, representer):


# def yaml_object(yml):
#     # type: (Any) -> Any
#     """decorator for classes that needs to dump/load objects
#     The tag for such objects is taken from the class attribute yaml_tag (or the
#     class name in lowercase in case unavailable)
#     If methods to_yaml and/or from_yaml are available, these are called for dumping resp.
#     loading, default routines (dumping a mapping of the attributes) used otherwise.
#     """

#     def yo_deco(cls):
#         # type: (Any) -> Any
#         tag = getattr(cls, "yaml_tag", "!" + cls.__name__)
#         try:
#             yml.representer.add_representer(cls, cls.to_yaml)
#             yml.representer.add_multi_representer(cls, cls.to_yaml)
#         except AttributeError:

#             def t_y(representer, data):
#                 # type: (Any, Any) -> Any
#                 return representer.represent_yaml_object(
#                     tag, data, cls, flow_style=representer.default_flow_style
#                 )

#             yml.representer.add_representer(cls, t_y)
#             # yml.representer.add_multi_representer(cls, t_y)
#         try:
#             yml.constructor.add_constructor(tag, cls.from_yaml)
#         except AttributeError:

#             def f_y(constructor, node):
#                 # type: (Any, Any) -> Any
#                 return constructor.construct_yaml_object(node, cls)

#             yml.constructor.add_constructor(tag, f_y)
#         return cls

#     return yo_deco
