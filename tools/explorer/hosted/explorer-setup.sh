#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

mkdir -p /srv
cd /srv
if [ -d "tt-mlir" ]; then
  cd tt-mlir
  git checkout main
  git pull
else
  git clone https://github.com/tenstorrent/tt-mlir.git tt-mlir
fi
cronexist=$(crontab -l | grep explorer-setup.sh || true)
if [ -z "$cronexist" ]; then
  echo "Setting up cron job for daily Explorer setup..."
  (crontab -l ; echo "0 5 * * * /bin/bash /srv/tt-mlir/tools/explorer/hosted/explorer-setup.sh >> /var/log/explorer-setup.log 2>&1") | crontab -
else
  echo "Cron job for Explorer setup already exists."
fi
cd tt-mlir/tools/explorer/hosted
echo "Building Explorer dockers..."
make build-full
echo "(Re)Starting Explorer services..."
make restart
echo "Explorer setup complete."
