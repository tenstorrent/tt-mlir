#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg $1: path to pytest test files
# arg $2: pytest marker expression to select tests to run
# arg $3: additional flags for pytest

set -e -o pipefail

TTRT_ARGS=""
PYTEST_ARGS=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
[[ "$RUNS_ON" == "p150" ]] && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"

for flag in $3; do
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH
