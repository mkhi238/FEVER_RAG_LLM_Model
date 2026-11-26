---
name: unit-tests
description: 'Unit tests: answers "is the code correct?". Runs the fast, pass/fail pytest checks in tests/ (config, retrievers, evidence scoring, FEVER metric, API, aggregator, store; no GPU, no big artifacts needed), all of them or one file, a -k name pattern, or the first N ("/unit-tests 10"), and reports passes and failures with file:line. The same tests run in GitHub Actions CI on every push (.github/workflows/unit-tests.yml). Use after any code change, or when asked to run, check or try the tests. It does NOT measure accuracy: that is /fever-full-test.'
argument-hint: "[N | file | -k pattern]"
---

# /unit-tests: run the unit tests

## Setup (always)

```bash
cd /mnt/d/FEVER_RAG_LLM_Model && source .venv/bin/activate && source scripts/env.sh
```

pytest uses `--capture=sys` from `pyproject.toml`; keep it, because the default tmpfile capture breaks on drvfs (D:).
The suite is slow because D: is a WSL drive, so pick a subset when the user asks for one.

## Pick what to run from the argument

| Argument | Command |
|---|---|
| none | `python -m pytest -q` |
| a number N (e.g. `10`) | `python -m pytest -q --co tests/ \| grep '::' \| head -N \| xargs python -m pytest -q` |
| a file or name (e.g. `scoring`, `tests/test_api.py`) | `python -m pytest -q tests/test_<name>.py` |
| `-k pattern` | `python -m pytest -q -k "<pattern>"` |
| a node id (`tests/x.py::test_y`) | `python -m pytest -q <node id>` |

For a number, collect first and show the user which N tests were picked. If collection fails (import error), report that
error; don't try to run anything.

Run with `nice -n 19` if the user's RL training may be on the machine. Don't add `-x` unless asked.

## Report

- One line: `X passed, Y failed, Z skipped in T s`.
- For each failure: the test id, the assertion or exception (one or two lines) and the file:line it points at.
- Don't fix failures inside this skill. Offer to (the `fix-fever` agent handles fixes).

## Notes

- Test files: aggregation, aggregator_llm_output, api, benchmark, config, faiss_indexer, fever_metric, hybrid_retriever,
  imports, instruct_model, posttrain_store, rear_aggregator, rerank, scoring, store (plus the deselected fever_suite).
- `claimscan/llm/` has no tests yet. `test_rear_aggregator.py` skips until `claimscan/instruct/rear_aggregator.py` exists.
- Tests must not need the GPU or the 4 GB corpus. If one tries to load them, say so in the report.
- `tests/test_fever_suite.py` (the FEVER claim suite) is deselected from a plain `pytest` by tests/conftest.py.
  Run it with `/fever-full-test [N]`, not this skill.

## CI (GitHub Actions)

`.github/workflows/unit-tests.yml` runs this same `python -m pytest -q` on every push and pull request
(ubuntu, Python 3.12, CPU torch, Java 21, `requirements-dev.txt`). There are no artifacts there
(`CLAIMSCAN_ARTIFACTS=/tmp/no-artifacts`) and `HF_HUB_OFFLINE=1`, so a new test that needs the artifacts or a model
download must skip itself (`pytest.mark.skipif(not config.artifacts_root.exists(), ...)`), or CI goes red.
Latest runs (public repo, no login): `curl -s "https://api.github.com/repos/mkhi238/FEVER_RAG_LLM_Model/actions/runs?per_page=5" | python3 -c "import json,sys; [print(r['head_branch'], r['head_sha'][:7], r['status'], r['conclusion'], r['html_url']) for r in json.load(sys.stdin)['workflow_runs']]"`
