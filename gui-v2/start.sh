#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/gui/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo ""
echo "  LLM Lab GUI v2"
echo "  ─────────────────────────────────"
echo ""

# ── Backend ───────────────────────────────────────────────────────────────────

# Check if backend is already running on port 8000
if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
  echo "  [backend]  Already running on port 8000 — skipping"
  BACKEND_PID=""
else
  # Set up venv if needed
  if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo "  [backend]  Creating virtual environment…"
    python3 -m venv "$BACKEND_DIR/venv"
  fi

  # Install dependencies
  echo "  [backend]  Installing dependencies…"
  "$BACKEND_DIR/venv/bin/pip" install -q -r "$BACKEND_DIR/requirements.txt"

  echo "  [backend]  Starting on port 8000…"
  cd "$BACKEND_DIR"
  source venv/bin/activate
  python main.py &
  BACKEND_PID=$!
  cd "$SCRIPT_DIR"

  # Wait for backend to be ready
  echo -n "  [backend]  Waiting for API"
  for i in $(seq 1 20); do
    if curl -sf http://localhost:8000/api/health > /dev/null 2>&1; then
      echo " — ready"
      break
    fi
    echo -n "."
    sleep 1
  done
fi

# ── Frontend ──────────────────────────────────────────────────────────────────

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "  [frontend] Installing npm dependencies…"
  cd "$FRONTEND_DIR"
  npm install
  cd "$SCRIPT_DIR"
fi

echo "  [frontend] Starting dev server on port 5174…"
cd "$FRONTEND_DIR"
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "  ─────────────────────────────────"
echo "  Backend API:  http://localhost:8000"
echo "  Frontend GUI: http://localhost:5174"
echo "  ─────────────────────────────────"
echo "  Press Ctrl+C to stop all servers"
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────────

function cleanup() {
  echo ""
  echo "  Shutting down…"
  if [ -n "$FRONTEND_PID" ]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  exit 0
}

trap cleanup INT TERM

wait
