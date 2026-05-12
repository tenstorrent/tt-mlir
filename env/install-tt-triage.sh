#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

PROJECT_ROOT=$(git rev-parse --show-toplevel)
TT_METAL_VERSION=$(grep 'set(TT_METAL_VERSION' $PROJECT_ROOT/third_party/CMakeLists.txt | sed 's/.*"\(.*\)".*/\1/')
mkdir temp
cd temp

wget -O requirements.txt "https://raw.githubusercontent.com/tenstorrent/tt-metal/${TT_METAL_VERSION}/tools/triage/requirements.txt"

pip install --no-cache-dir -r requirements.txt

cd ..
rm -rf ./temp
