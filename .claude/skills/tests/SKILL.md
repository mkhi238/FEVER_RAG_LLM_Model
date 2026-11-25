---
name: tests
description: Run the FEVER/ClaimScan repo's unit tests (pytest in tests/), either all of them, one file, tests matching a name, or just the first N (e.g. "/tests 10"), and report what passed and failed. Use when asked to run, check or try the unit tests.
argument-hint: "[N | file | -k pattern]"
---

# /tests: run the unit tests

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

- Test files: aggregation, api, benchmark, config, faiss_indexer, fever_metric, hybrid_retriever, imports, rear_aggregator,
  rerank, scoring, store, verifier_data (plus the deselected fever_suite).
- `claimscan/llm/` has no tests yet. `test_rear_aggregator.py` skips until `claimscan/instruct/rear_aggregator.py` exists.
- Tests must not need the GPU or the 4 GB corpus. If one tries to load them, say so in the report.
- `tests/test_fever_suite.py` (the FEVER claim suite) is deselected here by `-m "not fever_suite"` in pyproject.
  Run it with `/fever-300` or `/fever-10`, not this skill.
