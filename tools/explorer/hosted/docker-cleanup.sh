#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Get available disk space in GB
AVAILABLE_SPACE=$(df / | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))

echo "Available disk space: ${AVAILABLE_GB}GB"

# Check if available space is less than 50GB
if [ "$AVAILABLE_GB" -lt 50 ]; then
    echo "Available space is less than 50GB. Running docker system prune..."
    docker system prune -af --volumes
    echo "Docker cleanup completed."
else
    echo "Sufficient disk space available. No cleanup needed."
fi
