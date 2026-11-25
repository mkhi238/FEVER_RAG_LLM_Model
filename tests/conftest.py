def pytest_addoption(parser):
  parser.addoption("--fever-claims", type=int, default=300,
                   help="claims for the fever_suite test (300 = full benchmark sample, 10 = fast slice)")
  parser.addoption("--verifier", default="ours",
                   help="verifier for the gold-evidence test: 'ours' (the trained checkpoint) or a HuggingFace NLI model")
  parser.addoption("--gold-split", default="validation", choices=["validation", "train"],
                   help="claims for the gold-evidence test: 'validation' (the suite's) or 200 TRUE/FALSE train claims")


def pytest_collection_modifyitems(config, items):
  """The slow FEVER suite runs only when tests/test_fever_suite.py is named (or with -m fever_suite)."""
  if any("test_fever_suite" in a for a in config.args) or "fever_suite" in (config.getoption("-m") or ""):
    return
  keep = [i for i in items if "fever_suite" not in i.keywords]
  if len(keep) < len(items):
    config.hook.pytest_deselected(items=[i for i in items if "fever_suite" in i.keywords])
    items[:] = keep
