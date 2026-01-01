#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

GRAFANA_URL="${GRAFANA_URL:-https://grafana-cuda-demo.josnelihurt.me}"
GRAFANA_USER="${GRAFANA_ADMIN_USER:-admin}"
GRAFANA_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}"
SECRETS_FILE="${SECRETS_FILE:-.secrets/production.env}"

echo "Generating Grafana API token..."

if [ ! -f "$SECRETS_FILE" ]; then
    echo "Error: Secrets file not found: $SECRETS_FILE"
    exit 1
fi

echo "Waiting for Grafana to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s -f -o /dev/null "${GRAFANA_URL}/api/health"; then
        echo "Grafana is ready"
        break
    fi
    attempt=$((attempt + 1))
    echo "Waiting for Grafana... (attempt $attempt/$max_attempts)"
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "Error: Grafana did not become ready in time"
    exit 1
fi

echo "Creating or finding service account..."
SERVICE_ACCOUNT_RESPONSE=$(curl -s -X POST "${GRAFANA_URL}/api/serviceaccounts" \
    -H "Content-Type: application/json" \
    -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
    -d '{
        "name": "MCP Server",
        "role": "Admin",
        "isDisabled": false
    }' 2>/dev/null)

SERVICE_ACCOUNT_ID=$(echo "$SERVICE_ACCOUNT_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

if [ -z "$SERVICE_ACCOUNT_ID" ]; then
    echo "Searching for existing service account..."
    SEARCH_RESPONSE=$(curl -s "${GRAFANA_URL}/api/serviceaccounts/search" \
        -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" 2>/dev/null)
    SERVICE_ACCOUNT_ID=$(echo "$SEARCH_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
    
    if [ -z "$SERVICE_ACCOUNT_ID" ]; then
        echo "Error: Failed to create or find service account"
        echo "Response: $SERVICE_ACCOUNT_RESPONSE"
        exit 1
    fi
fi

echo "Creating token for service account $SERVICE_ACCOUNT_ID..."
TOKEN_RESPONSE=$(curl -s -X POST "${GRAFANA_URL}/api/serviceaccounts/${SERVICE_ACCOUNT_ID}/tokens" \
    -H "Content-Type: application/json" \
    -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
    -d '{
        "name": "MCP Server Token",
        "secondsToLive": 0
    }')

TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"key":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    if echo "$TOKEN_RESPONSE" | grep -q "already exists"; then
        echo "Token already exists, listing existing tokens..."
        TOKENS_RESPONSE=$(curl -s "${GRAFANA_URL}/api/serviceaccounts/${SERVICE_ACCOUNT_ID}/tokens" \
            -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}")
        
        TOKEN_ID=$(echo "$TOKENS_RESPONSE" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)
        
        if [ -n "$TOKEN_ID" ]; then
            echo "Deleting existing token $TOKEN_ID..."
            curl -s -X DELETE "${GRAFANA_URL}/api/serviceaccounts/${SERVICE_ACCOUNT_ID}/tokens/${TOKEN_ID}" \
                -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" > /dev/null
            
            echo "Creating new token..."
            TOKEN_RESPONSE=$(curl -s -X POST "${GRAFANA_URL}/api/serviceaccounts/${SERVICE_ACCOUNT_ID}/tokens" \
                -H "Content-Type: application/json" \
                -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
                -d '{
                    "name": "MCP Server Token",
                    "secondsToLive": 0
                }')
            
            TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"key":"[^"]*' | cut -d'"' -f4)
        fi
    fi
    
    if [ -z "$TOKEN" ]; then
        echo "Error: Failed to extract token from response"
        echo "Response: $TOKEN_RESPONSE"
        exit 1
    fi
fi

echo "Token generated successfully"

if grep -q "^GRAFANA_API_TOKEN=" "$SECRETS_FILE"; then
    sed -i "s|^GRAFANA_API_TOKEN=.*|GRAFANA_API_TOKEN=${TOKEN}|" "$SECRETS_FILE"
    echo "Updated existing GRAFANA_API_TOKEN in $SECRETS_FILE"
else
    echo "" >> "$SECRETS_FILE"
    echo "# Grafana API Token for MCP Server" >> "$SECRETS_FILE"
    echo "GRAFANA_API_TOKEN=${TOKEN}" >> "$SECRETS_FILE"
    echo "Added GRAFANA_API_TOKEN to $SECRETS_FILE"
fi

chmod 600 "$SECRETS_FILE"
echo "Done. Token saved to $SECRETS_FILE"

