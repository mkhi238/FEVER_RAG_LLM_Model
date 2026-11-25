def pytest_addoption(parser):
  parser.addoption("--fever-claims", type=int, default=300,
                   help="claims for the fever_suite test (300 = full benchmark sample, 10 = fast slice)")
