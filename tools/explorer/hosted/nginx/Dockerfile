# Tenstorrent tt-explorer Nginx Dockerfile
# Multi-stage build for Nginx with ACME module
# Based on: https://christiangn.medium.com/testing-the-native-nginx-http-01-acme-module-5109b42bafe4

# [Stage 1] Build the dynamic Nginx module - ngx_http_acme_module.so
FROM rust:1.89-bookworm AS nginx-http-acme-mod

RUN apt-get update && apt-get install --yes --no-install-recommends --no-install-suggests \
        libclang-dev \
        libpcre2-dev \
        libssl-dev \
        zlib1g-dev \
        pkg-config \
        git \
        grep \
        gawk \
        gnupg2 \
        sed \
        make \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Clone nginx and nginx-acme repositories
RUN git clone https://github.com/nginx/nginx.git /nginx \
 && git clone https://github.com/nginx/nginx-acme.git /mod

WORKDIR /nginx

# Checkout specific nginx version
RUN git checkout release-1.28.0

# Configure and build the ACME module
RUN auto/configure \
    --with-compat \
    --with-http_ssl_module \
    --add-dynamic-module=/mod \
    && make modules

# [Stage 2] Build the release image with the added ACME module
FROM nginx:1.28-bookworm AS nginx-acme

LABEL org.opencontainers.image.authors="it@tenstorrent.com"
LABEL maintainer="Tenstorrent IT <it@tenstorrent.com>"
LABEL description="Nginx with native ACME module support for tt-explorer"

# Install required runtime dependencies
RUN apt-get update && apt-get install --yes --no-install-recommends --no-install-suggests \
        libssl-dev \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

# Copy nginx HTTP ACME module to release image
COPY --from=nginx-http-acme-mod /nginx/objs/ngx_http_acme_module.so /usr/lib/nginx/modules/

# Copy main nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy custom nginx configuration template
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Expose HTTP and HTTPS ports
EXPOSE 80/tcp 443/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
