#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

GRAFANA_URL="${GRAFANA_URL:-https://grafana-cuda-demo.josnelihurt.me}"
SECRETS_FILE="${SECRETS_FILE:-.secrets/production.env}"
MCP_DIR="${MCP_DIR:-$HOME/.local/share/grafana-mcp}"

if [ -f "$SECRETS_FILE" ] && [ -z "$GRAFANA_API_TOKEN" ]; then
    source "$SECRETS_FILE"
fi

export GRAFANA_URL="${GRAFANA_URL:-https://grafana-cuda-demo.josnelihurt.me}"
export GRAFANA_API_TOKEN="${GRAFANA_API_TOKEN:-}"

if [ -z "$GRAFANA_API_TOKEN" ]; then
    echo "Error: GRAFANA_API_TOKEN not set. Please set it in environment or $SECRETS_FILE" >&2
    exit 1
fi

export GRAFANA_CLUSTER_URLS="demo=${GRAFANA_URL}"
export GRAFANA_API_TOKENS="demo=${GRAFANA_API_TOKEN}"
export GRAFANA_READ_ACCESS_TAGS=""
export GRAFANA_ROOT_FOLDER="/"
export GRAFANA_WRITE_ACCESS_TAGS="MCP"

if [ ! -d "$MCP_DIR" ]; then
    echo "Error: grafana-mcp not installed at $MCP_DIR" >&2
    echo "Please install it by running:" >&2
    echo "  git clone https://github.com/christian-schlichtherle/grafana-mcp.git $MCP_DIR" >&2
    echo "  cd $MCP_DIR && python3 -m pip install --user httpx 'mcp[cli]'" >&2
    exit 1
fi

if [ ! -f "$MCP_DIR/main.py" ]; then
    echo "Error: main.py not found in $MCP_DIR" >&2
    exit 1
fi

cd "$MCP_DIR"
exec python3 main.py

