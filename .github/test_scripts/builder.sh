#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# path: path to pytest test files
# args: pytest marker expression to select tests to run
# flags: "run-ttrt" or predefined additional flags for pytest and ttrt

runttrt=""
TTRT_ARGS=""
PYTEST_ARGS=""
for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "disable-eth-dispatch" ]] && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
    [[ "$flag" == "require-exact-mesh" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
done

pytest $1 -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH
if [[ "$runttrt" == "1" ]]; then
    ttrt run $TTRT_ARGS ttir-builder-artifacts/
    ttrt run $TTRT_ARGS stablehlo-builder-artifacts/
fi
