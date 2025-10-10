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

- **HTTPS (recommended):** https://localhost
- **HTTP (redirects to HTTPS):** http://localhost  
- **Direct Go server:** http://localhost:8080

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
Browser (https://localhost:443)
          ↓
    Caddy (reverse proxy)
          ↓
  Go Server (http://localhost:8080)
          ↓
    C++/CUDA Processing
```

## Files

- `Caddyfile` - Caddy configuration
- `.secrets/` - SSL certificates (not in git)
- `scripts/setup-ssl.sh` - Certificate generation
- `scripts/start-dev.sh` - Development environment starter

## Troubleshooting

**Port 443 already in use:**
```bash
sudo lsof -i :443
sudo systemctl stop caddy  # if running as service
```

**Certificate errors:**
- Regenerate certificates: `cd .secrets && rm *.pem && cd .. && ./scripts/setup-ssl.sh`

**Caddy not starting:**
- Check logs: `cat caddy.log`
- Verify certificates exist: `ls -la .secrets/*.pem`
