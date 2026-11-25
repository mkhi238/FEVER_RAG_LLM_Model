#!/usr/bin/env bash
# Wait for the ClaimScan api container to be healthy, then smoke-test it.
# Adapted from wrsmith108/docker-claude-skill's health script: wait on the Docker healthcheck, fall back to a direct probe.
#
#   .claude/skills/dockerfile/docker-health.sh [max_wait_s=420] [port=8000]
#
# Run from the repo root (where docker-compose.yml is). Exit 0 = healthy and all smoke claims answered.
set -u
MAX=${1:-420}
PORT=${2:-${API_PORT:-8000}}
URL="http://localhost:$PORT"

check_docker() { docker info >/dev/null 2>&1 || { echo "FAIL: Docker daemon not reachable (start Docker Desktop, enable WSL integration)"; exit 2; }; }
cid() { docker compose ps -q api 2>/dev/null; }
health() { docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$(cid)" 2>/dev/null; }
probe() { curl -sf "$URL/health" 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('models_loaded') else 1)" 2>/dev/null; }

check_docker
[ -n "$(cid)" ] || { echo "FAIL: api container not running (docker compose up -d api)"; exit 2; }

echo "waiting up to ${MAX}s for the api (models load in ~1-5 min; the healthcheck start period is 300 s)"
t0=$(date +%s); status=""
while :; do
  status=$(health)
  [ "$status" = "healthy" ] && break
  probe && { status="responding (healthcheck still in its start period)"; break; }
  if [ "$status" = "exited" ] || [ "$status" = "dead" ]; then
    echo "FAIL: container $status. Last logs:"; docker compose logs --tail 30 api; exit 1
  fi
  [ $(( $(date +%s) - t0 )) -ge "$MAX" ] && { echo "FAIL: not healthy after ${MAX}s (status: $status)"; docker compose logs --tail 30 api; exit 1; }
  sleep 5
done
echo "api: $status after $(( $(date +%s) - t0 ))s"
curl -s "$URL/health"; echo

fail=0
for claim in "Tokyo is the capital of Japan." "The Eiffel Tower is located in Berlin." "Barack Obama owned a purple bicycle as a child."; do
  t=$(date +%s.%N)
  body=$(curl -s -w '\n%{http_code}' -X POST "$URL/fact-check" -H 'content-type: application/json' \
         -d "{\"claim\": \"$claim\"}")
  code=$(echo "$body" | tail -1); json=$(echo "$body" | sed '$d')
  secs=$(python3 -c "import time; print(f'{time.time()-$t:.1f}')")
  summary=$(echo "$json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
ev = d.get('evidence') or []
src = ev[0].get('source', ev[0]) if ev and isinstance(ev[0], dict) else (ev[0] if ev else '-')
print(f\"{d.get('prediction')}  conf={d.get('confidence')}  evidence={len(ev)} first={src}  aggregator_error={d.get('aggregator_error')}\")
" 2>/dev/null || echo "unparseable response")
  echo "[$code] ${secs}s  $claim  ->  $summary"
  [ "$code" = "200" ] || fail=1
done

echo "user in container: uid $(docker compose exec -T api id -u)"
docker compose exec -T api sh -c 'touch /artifacts/.rw_test 2>/dev/null && echo "WARN: /artifacts is WRITABLE" || echo "/artifacts read-only: ok"'
docker stats --no-stream --format 'RAM {{.MemUsage}}  CPU {{.CPUPerc}}' "$(cid)"
exit $fail
