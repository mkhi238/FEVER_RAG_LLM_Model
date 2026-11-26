def pytest_addoption(parser):
  parser.addoption("--fever-claims", type=int, default=300,
                   help="claims for the fever_suite test (300 = full benchmark sample, 10 = fast slice)")
  parser.addoption("--verifier", default=None,
                   help="verifier for the FEVER suite: default verifier.inference_model from the config, "
                        "or another HuggingFace NLI model")
  parser.addoption("--aggregator", default="aggregator_dpo",
                   help="rear aggregator for the FEVER suite: an adapter dir under <artifacts>/models (default: the "
                        "trained DPO one) whose labels are scored, or 'none' for the verifier's labels")
  parser.addoption("--gold-split", default="validation", choices=["validation", "train"],
                   help="claims for the gold-evidence test: 'validation' (the suite's) or 200 TRUE/FALSE train claims")


def _deselect(config, items, marker):
  drop = [i for i in items if marker in i.keywords]
  if drop:
    config.hook.pytest_deselected(items=drop)
    items[:] = [i for i in items if marker not in i.keywords]


def pytest_collection_modifyitems(config, items):
  """The slow FEVER suite runs only when tests/test_fever_suite.py is named (or with -m fever_suite).
  Its gold-evidence measurement (~200 claims) is not part of /fever: it runs only when
  asked for with -k gold_evidence or -m gold_evidence."""
  asked = (config.getoption("-k") or "") + " " + (config.getoption("-m") or "")
  if "gold_evidence" not in asked:
    _deselect(config, items, "gold_evidence")
  if not (any("test_fever_suite" in a for a in config.args) or "fever_suite" in (config.getoption("-m") or "")):
    _deselect(config, items, "fever_suite")


import pytest  # noqa: E402


@pytest.fixture
def veto_on(monkeypatch):
  """The veto-gate tests: switch the rear aggregator's veto back on (it is off by default, rear_aggregator.VETO_ENABLED)."""
  ra = pytest.importorskip("claimscan.instruct.rear_aggregator")
  monkeypatch.setattr(ra, "VETO_ENABLED", True)
