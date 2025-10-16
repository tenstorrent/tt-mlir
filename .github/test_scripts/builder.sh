#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg $1: path to pytest test files
# arg $2: pytest marker expression to select tests to run
# arg $3: "run-ttrt" or predefined additional flags for pytest

set -e -o pipefail

runttrt=""
TTRT_ARGS=""
PYTEST_ARGS=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
[[ "$RUNS_ON" == "p150" ]] && PYTEST_ARGS="$PYTEST_ARGS --disable-eth-dispatch" && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"

for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH

if [[ "$runttrt" == "1" ]]; then
  if [ -d "ttir-builder-artifacts/emitpy" ]; then
        ttrt emitpy $TTRT_ARGS ttir-builder-artifacts/emitpy/
        cp emitpy_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_emitpy_${TEST_REPORT_PATH##*_} || true
  fi
  if [ -d "stablehlo-builder-artifacts/emitpy" ]; then
      ttrt emitpy $TTRT_ARGS stablehlo-builder-artifacts/emitpy/
      cp emitpy_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
      cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_stablehlo_emitpy_${TEST_REPORT_PATH##*_} || true
  fi
fi
