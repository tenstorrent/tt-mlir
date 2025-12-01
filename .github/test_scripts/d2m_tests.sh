#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi

for strategy in "basic" "greedy" "chaitin-briggs"; do
    echo "Running D2M Python tests with DST_ALLOCATION_STRATEGY=$strategy"
    DST_ALLOCATION_STRATEGY="$strategy" pytest -v -k "ttmetal" --junit-xml="$TEST_REPORT_PATH" test/python/golden
done
