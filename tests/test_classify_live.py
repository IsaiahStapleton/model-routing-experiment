"""Live test against the real 1.7b tier. Skipped if the tier is not up."""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request

import pytest

from router.classify import classify

API_BASE = "http://127.0.0.1:4000/v1"


def _key():
    env = os.path.expanduser("~/isaiah-routing-lab/.env")
    with open(env) as fh:
        match = re.search(r"(?<=LITELLM_MASTER_KEY=).*", fh.read())
    return match.group(0).strip()


def _tier_up():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8004/health", timeout=3):
            return True
    except Exception:
        return False


@pytest.mark.skipif(not _tier_up(), reason="qwen3-1.7b tier not running")
def test_live_classifier_returns_score_in_range():
    score = classify("What is 2 + 2?", [], API_BASE, _key(), timeout=15.0)
    assert score is not None, "classifier returned no parseable score"
    assert 1 <= score <= 10


@pytest.mark.skipif(not _tier_up(), reason="qwen3-1.7b tier not running")
def test_live_classifier_rates_hard_above_trivial():
    easy = classify("Say hello.", [], API_BASE, _key(), timeout=15.0)
    hard = classify(
        "Derive the memory bandwidth bound for MoE decode and explain why "
        "observed throughput falls short of it.",
        ["read_file", "bash"], API_BASE, _key(), timeout=15.0,
    )
    assert easy is not None and hard is not None
    assert hard > easy, f"expected hard>{easy}, got {hard}"
