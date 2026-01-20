#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

TT_METAL_VERSION=$(grep 'set(TT_METAL_VERSION' third_party/CMakeLists.txt | sed 's/.*"\(.*\)".*/\1/')

mkdir temp
cd temp

wget -O install_debugger.sh "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/scripts/install_debugger.sh"
wget -O ttexalens_ref.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/scripts/ttexalens_ref.txt"
wget -O requirements.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/tools/triage/requirements.txt"

ls -l
./install_debugger.sh

pip install --no-cache-dir -r requirements.txt

cd ..
rm -rf ./temp
