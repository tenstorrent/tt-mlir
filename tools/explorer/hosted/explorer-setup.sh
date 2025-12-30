#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

echo "Setting up Tenstorrent Explorer Hosted App..."
echo "Date: $(date)"
mkdir -p /srv
cd /srv
if [ -d "tt-mlir" ]; then
  cd tt-mlir
  git checkout main
  git pull
else
  git clone https://github.com/tenstorrent/tt-mlir.git tt-mlir
fi

cronexist=$(crontab -l 2>/dev/null | grep explorer-setup.sh || true)
if [ -z "$cronexist" ]; then
  echo "Setting up cron job for daily Explorer setup..."
  (crontab -l  2>/dev/null; echo "0 9 * * * /bin/bash /srv/tt-mlir/tools/explorer/hosted/explorer-setup.sh >> /var/log/explorer-setup.log 2>&1") | crontab -
else
  echo "Cron job for Explorer setup already exists."
fi

cronexist=$(crontab -l 2>/dev/null | grep docker-cleanup.sh || true)
if [ -z "$cronexist" ]; then
  echo "Setting up cron job for daily Docker cleanup..."
  (crontab -l  2>/dev/null; echo "0 8 * * * /bin/bash /srv/tt-mlir/tools/explorer/hosted/docker-cleanup.sh >> /var/log/docker-cleanup.log 2>&1") | crontab -
else
  echo "Cron job for Docker cleanup already exists."
fi

cd /srv/tt-mlir/tools/explorer/hosted
echo "Building Explorer dockers..."
make build-full
echo "(Re)Starting Explorer services..."
make restart
echo "Explorer setup complete."
