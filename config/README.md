# Configuration Files

This directory contains configuration files for the observability stack and application services.

## Observability Stack

### Loki (Log Aggregation)

**File:** `loki-config.yaml`

Loki configuration for log storage and retention:
- Storage: Filesystem (in-memory for development)
- Retention: 744h (31 days)
- Schema: v13 with TSDB
- Compactor enabled for log cleanup

### Promtail (Log Collection)

**File:** `promtail-config.yaml`

Promtail scrapes logs from Docker containers:
- **Target:** `cuda-image-processor` container
- **Pipeline:** JSON parsing for zerolog logs
- **Labels extracted:** level, trace_id, span_id
- **Output format:** Structured with timestamp

### Grafana

#### Datasources (Auto-Provisioned)

**File:** `grafana-datasources.yaml`

Two datasources configured:
1. **Loki** (default)
   - URL: http://loki:3100
   - Derived field: trace_id → Jaeger link
   
2. **Jaeger**
   - URL: http://jaeger:16686
   - Trace-to-Logs: Configured for Loki integration

#### Dashboards (Auto-Provisioned)

**File:** `grafana-dashboards.yaml`

Dashboard provisioning configuration:
- Provider: File-based
- Path: /etc/grafana/dashboards
- Update interval: 10s
- UI updates: Allowed

**File:** `grafana-dashboard-logs.json`

Pre-configured dashboard with 7 panels:

1. **C++ Layer (CUDA/spdlog)** - Native C++ logs
2. **Go Layer - All Levels** - Complete Go logs with trace context
3. **Go Layer - Errors & Warnings** - Filtered critical logs
4. **Go Layer - Info** - Operational logs only
5. **Log Volume by Level** - Bar gauge visualization
6. **Logs Rate Over Time** - Time series metrics
7. **Recent Traces with Logs** - Clickable trace_id table

Dashboard features:
- UID: `cuda-multi-layer-logs`
- Auto-refresh: 5s
- Time range: Last 6h (configurable)
- Direct links from trace_id to Jaeger

## Deployment

### Development

```bash
docker compose -f docker-compose.dev.yml up -d jaeger loki promtail grafana
```

Access:
- Grafana: http://localhost:3001
- Dashboard: http://localhost:3001/d/cuda-multi-layer-logs
- Jaeger: http://localhost:16686
- Loki: http://localhost:3100

### Production

```bash
docker compose up -d
```

Access via Traefik:
- Grafana: https://your-domain/grafana
- Dashboard: https://your-domain/grafana/d/cuda-multi-layer-logs
- Jaeger: http://your-domain:16686 (or configure Traefik)

## Dashboard Provisioning

Grafana automatically loads the dashboard on startup:

1. `grafana-dashboards.yaml` tells Grafana where to find dashboards
2. `grafana-dashboard-logs.json` is mounted at `/etc/grafana/dashboards/`
3. Grafana scans every 10s and loads new/updated dashboards
4. Dashboard persists across container restarts (grafana-storage volume)

To update the dashboard:
1. Edit `config/grafana-dashboard-logs.json`
2. Restart Grafana: `docker compose restart grafana`
3. Changes appear within 10s

## Trace-to-Logs Correlation

### From Logs to Traces

In Loki logs, trace_id fields have clickable links:
- Query: `{container="cuda-image-processor"} | json`
- Click "View Trace" next to trace_id
- Opens trace in Jaeger

### From Traces to Logs

In Grafana (using Jaeger datasource):
- Search for a trace
- Click "Logs for this trace" button
- Shows all logs with matching trace_id

### Manual Correlation

Copy trace_id from Jaeger, then in Loki:
```
{container="cuda-image-processor"} | json | trace_id="<paste-trace-id-here>"
```

## Log Format

### C++ Layer (spdlog)
```
[2025-10-16 11:55:15.369] [info] Initializing CUDA context (device: 0)
```

### Go Layer (zerolog)
```json
{
  "level": "info",
  "trace_id": "b32004ea79b417471957d104cdf99740",
  "span_id": "93825b9a5735b667",
  "transport_format": "json",
  "time": "2025-10-16T11:55:15.809309798-07:00",
  "caller": "webserver/pkg/interfaces/websocket/handler.go:61",
  "message": "WebSocket connected"
}
```

## Maintenance

### Viewing Raw Config in Grafana

1. Navigate to Configuration → Data Sources
2. Check Loki and Jaeger settings
3. Verify "Derived Fields" in Loki (trace_id → Jaeger)

### Exporting Dashboard

If you make changes in Grafana UI:
1. Open dashboard
2. Settings (gear icon) → JSON Model
3. Copy JSON
4. Paste to `config/grafana-dashboard-logs.json`
5. Commit to repository

### Troubleshooting

Check provisioning logs:
```bash
docker logs grafana 2>&1 | grep provision
```

Verify files are mounted:
```bash
docker exec grafana ls -la /etc/grafana/provisioning/datasources/
docker exec grafana ls -la /etc/grafana/dashboards/
```

## Notes

- Dashboards are version-controlled in this repository
- No manual setup required on new deployments
- Dashboard updates via file changes (Infrastructure as Code)
- Grafana storage volume preserves user preferences and annotations

