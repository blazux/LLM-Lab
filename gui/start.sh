#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT/gui/backend"
FRONTEND_DIR="$ROOT/gui/frontend"
VENV="${LLMLAB_VENV:-$ROOT/.venv}"
API_PORT="${LLMLAB_API_PORT:-8000}"
UI_PORT="${LLMLAB_UI_PORT:-5174}"
API_URL="http://127.0.0.1:$API_PORT"
PIDS=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if curl --max-time 2 -sf "$API_URL/api/health" >/dev/null; then
  echo "Backend already running: $API_URL"
else
  bash "$ROOT/scripts/bootstrap.sh" --venv "$VENV"
  setsid "$VENV/bin/python" -m uvicorn main:app --app-dir "$BACKEND_DIR" --host 127.0.0.1 --port "$API_PORT" &
  BACKEND_PID=$!
  PIDS+=("$BACKEND_PID")
  READY=false
  for ((attempt=0; attempt<60; attempt++)); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then echo 'Backend exited during startup.' >&2; exit 1; fi
    if curl --max-time 1 -sf "$API_URL/api/health" >/dev/null; then READY=true; break; fi
    sleep 1
  done
  if [[ "$READY" != true ]]; then echo 'Backend did not become ready within 60 seconds.' >&2; exit 1; fi
fi
(cd "$FRONTEND_DIR" && npm ci)
setsid env VITE_API_TARGET="$API_URL" npm --prefix "$FRONTEND_DIR" run dev -- --host 127.0.0.1 --port "$UI_PORT" --strictPort &
PIDS+=("$!")
echo "API: $API_URL | GUI: http://127.0.0.1:$UI_PORT"
echo 'Press Ctrl+C to stop the servers started by this command.'
wait -n "${PIDS[@]}"
