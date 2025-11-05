#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi
export TT_EXPLORER_GENERATED_MLIR_TEST_DIRS=$BUILD_DIR/test/ttmlir/Silicon/TTNN/n150/perf,$BUILD_DIR/test/python/golden/ttnn
export TT_EXPLORER_GENERATED_TTNN_TEST_DIRS=$BUILD_DIR/test/python/golden/ttnn
# Set no_proxy to bypass proxy for localhost connections (needed for explorer tests in CIv2)
# See: https://github.com/tenstorrent/github-ci-infra/issues/1187
# Append to existing no_proxy to preserve infrastructure bypass rules (harbor, large-file-cache)
# Using lowercase only as that's all model-explorer needs (Python requests lib / urllib checks lowercase first then falls back to uppercase)
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
pytest -svv "$@" --junit-xml=$TEST_REPORT_PATH
