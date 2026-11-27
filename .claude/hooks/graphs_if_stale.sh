#!/usr/bin/env bash
# Stop hook: if any eval result is newer than graphs/README.md, rebuild graphs/ in the background.
# The check is a few `find` calls (milliseconds). The rebuild runs detached at nice 19, so the hook never blocks.
# Log: <artifacts>/eval/graphs_hook.log (kept out of graphs/ so that folder stays clean).
set -u
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 0

ART="${CLAIMSCAN_ARTIFACTS:-$(sed -n 's/^artifacts_root:[[:space:]]*//p' configs/default.yaml | head -1)}"
[ -d "$ART/eval" ] || exit 0
STAMP="graphs/README.md"

newer=$(find "$ART/eval/runs" -maxdepth 2 -name summary.json ${STAMP:+$( [ -e "$STAMP" ] && echo "-newer $STAMP")} 2>/dev/null | head -1)
[ -z "$newer" ] && newer=$(find "$ART/eval/e2e" -maxdepth 1 -name 'e2e_*.json' ${STAMP:+$( [ -e "$STAMP" ] && echo "-newer $STAMP")} 2>/dev/null | head -1)
[ -z "$newer" ] && exit 0

LOCK="$ART/eval/.graphs_hook.lock"
LOG="$ART/eval/graphs_hook.log"
# flock -n: if a rebuild is already running, do nothing
setsid nohup flock -n "$LOCK" bash -c '
  echo "[$(date "+%F %T")] rebuilding graphs (newer result: $1)"
  source .venv/bin/activate && source scripts/env.sh >/dev/null 2>&1
  nice -n 19 python -m claimscan.graph.make_graphs
  echo "[$(date "+%F %T")] done (exit $?)"
' _ "$newer" >>"$LOG" 2>&1 < /dev/null &

printf '{"systemMessage": "New eval results found: rebuilding graphs/ in the background (log: %s)"}\n' "$LOG"
exit 0
