# TT-Explorer Docker Compose Management

.PHONY: help build up down logs shell clean restart status local-up local-down local-logs local-build

# Default target
help:
	@echo "TT-Explorer Docker Compose Management"
	@echo ""
	@echo "Production (with OAuth2):"
	@echo "  setup     - Copy env.example to .env and show OAuth2 setup guide"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services (production with OAuth2)"
	@echo "  down      - Stop all services"
	@echo "  logs      - Show logs from all services"
	@echo ""
	@echo "Local Development (no OAuth2):"
	@echo "  local-up    - Start services locally without authentication"
	@echo "  local-down  - Stop local services"
	@echo "  local-logs  - Show logs from local services"
	@echo "  local-build - Build images for local development"
	@echo ""
	@echo "Utilities:"
	@echo "  shell       - Open shell in tt-explorer container"
	@echo "  oauth-shell - Open shell in oauth2-proxy container (production only)"
	@echo "  nginx-shell - Open shell in nginx container"
	@echo "  clean       - Remove containers, networks, and volumes"
	@echo "  restart     - Restart all services"
	@echo "  status      - Show status of all services"

# Setup environment file
setup:
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "Created .env file from env.example"; \
		echo ""; \
		echo "IMPORTANT: Configure before starting production services"; \
		echo "See README.md for deployment guide"; \
		echo "See nginx/oauth2-setup.md for OAuth2 setup instructions"; \
		echo ""; \
		echo "Required environment variables:"; \
		echo "  ACME Certificate (HTTP-01 challenge):"; \
		echo "    - ACME_DIRECTORY_URL (default: Let's Encrypt)"; \
		echo "    - ACME_CONTACT"; \
		echo "    - FQDN (must be publicly resolvable)"; \
		echo ""; \
		echo "  OAuth2 Authentication:"; \
		echo "    - OAUTH2_PROVIDER (azure, google, github, oidc)"; \
		echo "    - OAUTH2_CLIENT_ID"; \
		echo "    - OAUTH2_CLIENT_SECRET"; \
		echo "    - OAUTH2_COOKIE_SECRET (generate: python -c 'import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"; \
		echo "    - OAUTH2_AZURE_TENANT (if using Azure/Entra ID)"; \
		echo ""; \
		echo "Please edit .env with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Production targets
build:
	docker compose -p tt-explorer -f docker-compose.yml build

up:
	docker compose -p tt-explorer -f docker-compose.yml up -d

down:
	docker compose -p tt-explorer -f docker-compose.yml down

logs:
	docker compose -p tt-explorer -f docker-compose.yml logs -f

# Local development targets
local-build:
	docker compose -f docker-compose.local.yml build

local-up:
	@echo "Starting TT-Explorer locally (no authentication)..."
	@echo "Access at: http://localhost"
	docker compose -f docker-compose.local.yml up -d

local-down:
	docker compose -f docker-compose.local.yml down

local-logs:
	docker compose -f docker-compose.local.yml logs -f

# Shell access
shell:
	@docker compose -p tt-explorer -f docker-compose.yml ps -q explorer >/dev/null 2>&1 && docker compose -p tt-explorer -f docker-compose.yml exec explorer /bin/bash || \
	 docker compose -f docker-compose.local.yml exec explorer /bin/bash

oauth-shell:
	docker compose -p tt-explorer -f docker-compose.yml exec oauth2-proxy /bin/sh

nginx-shell:
	@docker compose -p tt-explorer -f docker-compose.yml ps -q nginx >/dev/null 2>&1 && docker compose -p tt-explorer -f docker-compose.yml exec nginx /bin/bash || \
	 docker compose -f docker-compose.local.yml exec nginx /bin/bash

# Clean up everything
clean:
	docker compose -p tt-explorer -f docker-compose.yml down -v --remove-orphans
	docker compose -f docker-compose.local.yml down -v --remove-orphans
	docker system prune -f

# Restart services
restart: down up

# Show service status
status:
	@echo "Production services:"
	@docker compose -p tt-explorer -f docker-compose.yml ps || echo "No production services running"
	@echo ""
	@echo "Local services:"
	@docker compose -f docker-compose.local.yml ps || echo "No local services running"