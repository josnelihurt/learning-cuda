# Secrets Directory

This directory contains SSL certificates and other sensitive files.

## Setup SSL Certificates

### Option 1: Using mkcert (Recommended for local development)

```bash
# Install mkcert (if not already installed)
# Ubuntu/Debian:
sudo apt install libnss3-tools
wget -O mkcert https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64
chmod +x mkcert
sudo mv mkcert /usr/local/bin/

# Install local CA
mkcert -install

# Generate certificates in this directory
cd .secrets
mkcert localhost 127.0.0.1 ::1

# This creates:
# - localhost+2.pem (certificate)
# - localhost+2-key.pem (private key)
```

### Option 2: Using OpenSSL (Self-signed)

```bash
cd .secrets
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout localhost-key.pem \
  -out localhost.pem \
  -subj "/C=US/ST=State/L=City/O=Dev/CN=localhost"
```

## File Structure

```
.secrets/
├── README.md              # This file (committed)
├── .gitignore            # Ignore all certificates (committed)
├── localhost+2.pem       # SSL certificate (not committed)
├── localhost+2-key.pem   # SSL private key (not committed)
└── example.pem.example   # Example file showing expected format
```

## Security Note

**Never commit actual certificate files to version control!**

All `.pem` and `.key` files in this directory are automatically ignored by git.

