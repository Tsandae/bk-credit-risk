import yaml, pathlib

CFG = yaml.safe_load((pathlib.Path(__file__).parents[1] / "config" / "features.yaml").read_text())


def test_no_leakage_in_features():
    banned = set(CFG["leakage"]) | set(CFG["exclude_identifiers"]) | set(CFG["benchmark_only"]) | set(CFG["monitor_only"])
    assert not (set(CFG["features"]) & banned)


def test_target_defined():
    assert CFG["target"] == "loan_status"
