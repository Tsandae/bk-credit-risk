import pandas as pd
import pytest

from pipeline import config
from pipeline.anonymize import anonymize, k_anonymity, pii_scan, pseudonymize, scrub_text
from pipeline.clean import clean


def test_scrub_text_masks_email_phone_ssn():
    out = scrub_text("mail a.b@example.com or call 555-123-4567, ssn 123-45-6789")
    assert "@" not in out and "555-123-4567" not in out and "123-45-6789" not in out


def test_pseudonymize_is_deterministic_and_salt_dependent():
    s = pd.Series(["1", "2", "1"])
    a, b = pseudonymize(s, "salt-a"), pseudonymize(s, "salt-b")
    assert a.iloc[0] == a.iloc[2] and a.iloc[0] != a.iloc[1]
    assert a.iloc[0] != b.iloc[0]


def test_anonymize_removes_pii_and_enforces_k(raw_df):
    out, report = anonymize(clean(raw_df))
    for col in config.DIRECT_PII:
        assert col not in out.columns
    assert "loan_key" in out.columns and out["loan_key"].is_unique
    assert report["residual_pii_patterns"] == 0
    assert report["k_after"] >= config.K_ANONYMITY
    assert k_anonymity(out) >= config.K_ANONYMITY


def test_raw_text_contains_pii_before_anonymisation(raw_df):
    assert pii_scan(raw_df[["desc"]]) > 0  # the synthetic data really does leak e-mails/phones


def test_anonymize_refuses_without_salt(raw_df, monkeypatch):
    monkeypatch.delenv("ANON_SALT", raising=False)
    with pytest.raises(RuntimeError):
        anonymize(clean(raw_df))
