#!/usr/bin/env bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "Starting Central CCTV Intelligence Server on 0.0.0.0:8000..."
exec .venv/bin/uvicorn server.backend.main:app --host 0.0.0.0 --port 8000

