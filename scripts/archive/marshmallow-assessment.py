from pprint import pprint

from marshmallow import Schema, fields, ValidationError


class UserSchema(Schema):
    name = fields.String(required=True)
    age = fields.Integer(missing="no age")


my_serialized_user = {"name": "Joe"}

schema = UserSchema()

my_user = schema.load(my_serialized_user)
pprint(my_user)
