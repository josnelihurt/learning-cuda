# HTTPS Setup for Local Development

This project is configured to run with HTTPS support using Caddy as a reverse proxy.

## Quick Start

### 1. Generate SSL Certificates (First time only)

```bash
# Certificates are auto-generated with OpenSSL
./scripts/setup-ssl.sh
```

This creates self-signed certificates in `.secrets/` directory.

### 2. Install Caddy (if not installed)

**Ubuntu/Debian:**
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

**Or download from:** https://caddyserver.com/download

### 3. Start Development Environment

**Option A: Automated (recommended)**
```bash
./scripts/start-dev.sh
```

**Option B: Manual**

Terminal 1 - Start Go server:
```bash
bazel run //webserver/cmd/server:server
```

Terminal 2 - Start Caddy:
```bash
caddy run
```

### 4. Access the Application

- **HTTPS (recommended):** https://localhost:8443
- **HTTP (redirects to HTTPS):** http://localhost:8000
- **Direct Go server:** http://localhost:8080

**Note:** Using port 8443 for HTTPS (doesn't require root permissions)

## Why HTTPS?

Modern browsers require HTTPS for accessing:
- 📷 Webcam/Camera (getUserMedia API)
- 🎤 Microphone
- 📍 Geolocation
- 🔐 Other secure features

## Certificate Trust

The self-signed certificates will show a browser warning. This is normal for development.

**To avoid warnings, use mkcert instead:**
```bash
# Install mkcert
wget -O /tmp/mkcert https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64
chmod +x /tmp/mkcert
sudo mv /tmp/mkcert /usr/local/bin/

# Generate trusted certificates
cd .secrets
mkcert localhost 127.0.0.1 ::1
```

## Architecture

```
Browser (https://localhost:8443)
          ↓
    Caddy (reverse proxy)
          ↓
  Go Server (http://localhost:8080)
          ↓
    C++/CUDA Processing
```

**Why port 8443?** Ports below 1024 (like 443) require root privileges on Linux. Port 8443 is a standard alternative HTTPS port that doesn't require special permissions.

## Files

- `Caddyfile` - Caddy configuration
- `.secrets/` - SSL certificates (not in git)
- `scripts/setup-ssl.sh` - Certificate generation
- `scripts/start-dev.sh` - Development environment starter

## Troubleshooting

**Port 8443 already in use:**
```bash
lsof -i :8443
pkill caddy  # if you have another Caddy running
```

**Want to use standard port 443?**
Give Caddy permission to bind to privileged ports:
```bash
sudo setcap cap_net_bind_service=+ep $(which caddy)
# Then update Caddyfile to use port 443
```

**Certificate errors:**
- Regenerate certificates: `cd .secrets && rm *.pem && cd .. && ./scripts/setup-ssl.sh`

**Caddy not starting:**
- Check logs: `cat caddy.log`
- Verify certificates exist: `ls -la .secrets/*.pem`
