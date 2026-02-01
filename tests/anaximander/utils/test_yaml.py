from anaximander.utils.yaml import nx_yaml_dump, nx_yaml_load


def test_nx_yaml_roundtrip_basic():
    payload = {"value": "ok", "num": 3, "flag": False, "none": None}
    dumped = nx_yaml_dump(payload)
    loaded = nx_yaml_load(dumped)
    assert loaded == payload
