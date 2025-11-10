# TT-Explorer - Hosted Deployment

This directory contains all files needed to deploy TT-Explorer as a hosted web service with OAuth2 authentication and automatic HTTPS certificate management.

TT-Explorer is a web-based visualization and debugging tool for TT-MLIR models.

Full documentation: [TT-Explorer docs](https://docs.tenstorrent.com/tt-mlir/tt-explorer/tt-explorer.html)

## Quick Start

All commands should be run from this `hosted/` directory:

```bash
cd tools/explorer/hosted
make help
```

## Deployment Modes

Two deployment modes are available via Docker Compose:

### Local Development (No Authentication)

For local development without OAuth2 authentication:

```bash
make local-up
```

Access at: `http://localhost`

No configuration required. Explorer runs on port 8080 (also exposed to host), proxied through nginx on port 80.

**Stop:**
```bash
make local-down
```

### Production Deployment (OAuth2 + HTTPS)

Production deployment with OAuth2 authentication and automatic HTTPS via ACME:

#### Prerequisites

1. **Public DNS**: FQDN must resolve to the server's public IP
2. **ACME Server**: HTTP-01 challenge capable ACME server (Let's Encrypt, HashiCorp Vault PKI, etc.)
3. **OAuth2 Provider**: Azure Entra ID, Google, GitHub, or generic OIDC provider
4. **Ports**: 80 and 443 accessible from the internet

#### Setup

1. **Configure environment:**
   ```bash
   make setup
   ```
   This creates `.env` from `env.example`. Edit `.env` with:

   **Required - ACME Certificate:**
   - `ACME_DIRECTORY_URL` - ACME directory endpoint (default: Let's Encrypt production)
   - `ACME_CONTACT` - Contact email for certificate notifications
   - `FQDN` - Your fully qualified domain name

   **Required - OAuth2 Authentication:**
   - `OAUTH2_PROVIDER` - Provider type: `azure`, `google`, `github`, `oidc`
   - `OAUTH2_CLIENT_ID` - OAuth2 client ID
   - `OAUTH2_CLIENT_SECRET` - OAuth2 client secret
   - `OAUTH2_COOKIE_SECRET` - 32-byte base64 secret (generate: `python -c 'import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'`)
   
   **For Azure Entra ID (additional):**
   - `OAUTH2_AZURE_TENANT` - Azure tenant ID

   **Optional:**
   - `OAUTH2_EMAIL_DOMAINS` - Restrict to specific email domains (default: `*`)
   - See `env.example` for additional options

2. **Configure OAuth2 provider:**
   
   See `nginx/oauth2-setup.md` for detailed OAuth2 provider setup (Azure, Google, GitHub, OIDC).
   
   Redirect URI must be: `https://{FQDN}/oauth2/callback`

3. **Deploy:**
   ```bash
   make build
   make up
   ```

Access at: `https://{FQDN}`

**Architecture:**
```
Internet → nginx:80/443 (ACME + HTTPS termination)
            ↓
         oauth2-proxy:4180 (authentication)
            ↓
         explorer:8080 (application)
```

Only nginx ports (80/443) are exposed to the host. Explorer and oauth2-proxy are internal to the Docker network.

**View logs:**
```bash
make logs
```

**Stop:**
```bash
make down
```

## ACME Certificate Servers

The production deployment uses nginx's native ACME module for automatic certificate provisioning via HTTP-01 challenge.

**Supported ACME servers:**
- **Let's Encrypt** (default)
  - Production: `https://acme-v02.api.letsencrypt.org/directory`
  - Staging: `https://acme-staging-v02.api.letsencrypt.org/directory`
- **HashiCorp Vault PKI**: `https://vault.example.com/v1/pki/acme/directory`
- **Smallstep**: `https://ca.example.com/acme/acme/directory`
- Any RFC 8555 compliant ACME server with HTTP-01 challenge support

Set `ACME_DIRECTORY_URL` in `.env` to your ACME server's directory endpoint.

## Makefile Commands

All commands are run from the `hosted/` directory:

```bash
make help              # Show all available commands

# Local Development
make local-up          # Start without authentication
make local-down        # Stop local services
make local-logs        # View local logs
make local-build       # Build local images

# Production
make setup             # Initialize .env configuration
make build             # Build production images
make up                # Start with OAuth2 + HTTPS
make down              # Stop production services
make logs              # View production logs

# Utilities
make shell             # Shell into explorer container
make nginx-shell       # Shell into nginx container
make oauth-shell       # Shell into oauth2-proxy container (production only)
make status            # Show status of all services
make clean             # Remove all containers, networks, volumes
```

## Directory Structure

```
tools/explorer/
├── Dockerfile.explorer   # Main application Dockerfile
└── hosted/               # All hosted deployment files
    ├── README.md         # This file
    ├── Makefile          # Deployment Makefile (run commands from here)
    ├── docker-compose.yml
    ├── docker-compose.local.yml
    ├── docker-entrypoint.sh
    ├── env.example
    └── nginx/
        ├── Dockerfile
        ├── nginx.conf
        ├── nginx.conf.template
        ├── nginx.local.conf
        └── oauth2-setup.md
```

## Troubleshooting

**ACME certificate issues:**
- Ensure FQDN resolves to server's public IP
- Verify port 80 is accessible (needed for HTTP-01 challenge)
- Check nginx logs: `docker logs tt-explorer-nginx`
- Test with Let's Encrypt staging first to avoid rate limits

**OAuth2 authentication issues:**
- Verify redirect URI matches exactly: `https://{FQDN}/oauth2/callback`
- Check oauth2-proxy logs: `docker logs tt-explorer-oauth2-proxy`
- Ensure cookie secret is properly base64 encoded (32 bytes)
- See `nginx/oauth2-setup.md` for provider-specific troubleshooting

## How It Works

The Docker build context is set to the parent directory (`tools/explorer/`) so that `Dockerfile.explorer` can access the full tt-mlir repository for building. The docker-compose files reference the parent directory as the build context (`context: ..`), ensuring the container has access to all necessary source files.

All deployment configuration (nginx, OAuth2, environment variables) is contained in this `hosted/` directory, keeping deployment concerns separate from the core application code.
