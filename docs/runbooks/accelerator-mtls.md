# Accelerator mTLS Runbook

The C++ accelerator dials outbound to the Go cloud server over a **mutual TLS** (mTLS)
connection. Both sides must be issued certificates signed by the shared dev CA.

## Quick-start (dev machine, one time)

```bash
# 1. Create the CA
./scripts/dev/mint-accelerator-ca.sh

# 2. Server cert — used by the Go control server (cloud side)
./scripts/dev/mint-accelerator-cert.sh --name accelerator-server --type server

# 3. Client cert — used by the C++ accelerator (home/Jetson side)
./scripts/dev/mint-accelerator-cert.sh --name dev-accelerator-client --type client
```

This writes to `.secrets/` which is already in `.gitignore`.

## Files produced

| File | Who uses it |
|------|-------------|
| `.secrets/accelerator-ca.pem` | Both sides — the trust root |
| `.secrets/accelerator-ca-key.pem` | CA only, never distribute |
| `.secrets/accelerator-server.pem` | Go server `tls.cert_file` |
| `.secrets/accelerator-server-key.pem` | Go server `tls.key_file` |
| `.secrets/dev-accelerator-client.pem` | C++ accelerator cert |
| `.secrets/dev-accelerator-client-key.pem` | C++ accelerator key |

## Go server config (`config/config.dev.yaml`)

```yaml
processor:
  listen_address: ":60062"
  tls:
    cert_file: ".secrets/accelerator-server.pem"
    key_file:  ".secrets/accelerator-server-key.pem"
    client_ca_file: ".secrets/accelerator-ca.pem"
```

## C++ accelerator flags (passed at runtime)

```
--control_addr=<go-server-ip>:60062
--client_cert=.secrets/dev-accelerator-client.pem
--client_key=.secrets/dev-accelerator-client-key.pem
--ca_cert=.secrets/accelerator-ca.pem
```

## Verifying a cert chain

```bash
openssl verify -CAfile .secrets/accelerator-ca.pem .secrets/accelerator-server.pem
openssl verify -CAfile .secrets/accelerator-ca.pem .secrets/dev-accelerator-client.pem
```

## Regenerating certs (e.g., expired)

Delete the old cert and re-run:

```bash
rm .secrets/accelerator-server.pem .secrets/accelerator-server-key.pem
./scripts/dev/mint-accelerator-cert.sh --name accelerator-server --type server
```

To rotate the CA entirely, delete all files and rerun both scripts in order.

## Jetson / remote machine setup

1. Copy the CA cert and client cert pair to the Jetson:

```bash
scp .secrets/accelerator-ca.pem .secrets/dev-accelerator-client.pem \
    .secrets/dev-accelerator-client-key.pem  jetson:~/cuda-learning/.secrets/
```

2. Start the accelerator binary with the correct flags pointing at your cloud server.
