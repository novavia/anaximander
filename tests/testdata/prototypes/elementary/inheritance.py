import anaximander as nx


@nx.compile("dataclasses")
class A(nx.Model):
    a: int = nx.field()


@nx.compile("dataclasses")
class B(A):
    b: int = nx.field()


# class Mixin:

#     @property
#     def a_plus_b(self):
#         return self.a + self.b


# class C(B, Mixin):
#     pass
