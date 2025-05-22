import yaml, pathlib, pprint

settings = yaml.safe_load(
    pathlib.Path("config/settings.yaml").read_text()
)
assert settings["version"] == "0.1", "version key missing/incorrect"
required_top = {"weights", "caps"}
assert required_top.issubset(settings), "weights or caps section missing"

# basic sanity: all weights are floats, caps are ints
for k, v in settings["weights"].items():
    assert isinstance(v, (int, float)), f"weight {k} not numeric"
for k, v in settings["caps"].items():
    assert isinstance(v, int), f"cap {k} not integer"

pprint.pprint(settings)
print("✅ settings.yaml looks good")
