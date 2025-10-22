# Secrets Directory

This directory contains SSL certificates and other sensitive files.

## Setup SSL Certificates

### Automatic Generation (Recommended)

SSL certificates are generated automatically when you run:

```bash
./scripts/dev/start.sh
```

The script will detect missing certificates and generate them using Docker (no local tools needed).

### Manual Generation

```bash
./scripts/docker/generate-certs.sh
```

This uses Docker with Alpine + OpenSSL to generate self-signed certificates.

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

