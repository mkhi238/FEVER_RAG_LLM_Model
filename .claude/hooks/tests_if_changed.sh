#!/usr/bin/env bash
# Stop hook: the hook form of `/tests 10`.
# If any .py under claimscan/ or tests/ (or pyproject.toml) changed since the last run, run the first 10 pytest
# tests in the background (detached, nice 19, flock so only one run at a time). The hook itself only does a
# `find` and returns at once. The next Stop after a run finishes reports its result line once.
# Log: <artifacts>/eval/tests_hook.log. Stamps: <artifacts>/eval/.tests_hook.{stamp,result,reported}.
set -u
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 0

ART="${CLAIMSCAN_ARTIFACTS:-$(sed -n 's/^artifacts_root:[[:space:]]*//p' configs/default.yaml | head -1)}"
[ -d "$ART/eval" ] || exit 0
N="${CLAIMSCAN_HOOK_TESTS:-10}"
STAMP="$ART/eval/.tests_hook.stamp"      # touched when a run starts
RESULT="$ART/eval/.tests_hook.result"    # pytest's last summary line, written when a run ends
REPORTED="$ART/eval/.tests_hook.reported"
LOCK="$ART/eval/.tests_hook.lock"
LOG="$ART/eval/tests_hook.log"

msg=""
# 1. Report a finished run once.
if [ -s "$RESULT" ] && { [ ! -e "$REPORTED" ] || [ "$RESULT" -nt "$REPORTED" ]; }; then
  msg="Hook /tests $N: $(cat "$RESULT") (log: $LOG)"
  touch "$REPORTED"
fi

# 2. Start a run if code changed since the last one.
if [ -e "$STAMP" ]; then
  changed=$(find claimscan tests pyproject.toml -name '*.py' -newer "$STAMP" -o -name pyproject.toml -newer "$STAMP" 2>/dev/null | head -1)
else
  changed="first run"
fi
if [ -n "$changed" ] && flock -n "$LOCK" true; then
  touch "$STAMP"
  setsid nohup flock -n "$LOCK" bash -c '
    N=$1; RESULT=$2
    echo "[$(date "+%F %T")] running first $N tests (changed: $3)"
    source .venv/bin/activate && source scripts/env.sh >/dev/null 2>&1
    ids=$(nice -n 19 python -m pytest -q --co tests/ 2>&1 | grep "::" | head -"$N")
    if [ -z "$ids" ]; then
      echo "collection failed"; nice -n 19 python -m pytest -q --co tests/ 2>&1 | tail -20
      echo "collection FAILED (import error?)" > "$RESULT"; exit 1
    fi
    echo "$ids"
    out=$(nice -n 19 python -m pytest -q $ids 2>&1); rc=$?
    echo "$out"
    { echo "$out" | grep -E "passed|failed|error" | tail -1; } > "$RESULT"
    [ $rc -ne 0 ] && echo "$out" | grep -E "^FAILED|^ERROR" | head -5 | sed "s/^/  /" >> "$RESULT"
    echo "[$(date "+%F %T")] done (exit $rc)"
  ' _ "$N" "$RESULT" "$changed" >>"$LOG" 2>&1 < /dev/null &
  msg="${msg:+$msg; }Code changed: running first $N tests in the background (log: $LOG)"
fi

[ -n "$msg" ] && python3 -c 'import json,sys; print(json.dumps({"systemMessage": sys.argv[1]}))' "$msg"
exit 0
