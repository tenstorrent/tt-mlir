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
FLATBUFFER=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
[[ "$RUNS_ON" == "p150" ]] && PYTEST_ARGS="$PYTEST_ARGS --disable-eth-dispatch" && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"

for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=${TEST_REPORT_PATH%_*}_builder_${TEST_REPORT_PATH##*_}

# Messy file management temporary until issue #5309 is resolved
if [[ "$runttrt" == "1" ]]; then
    if [ -d ttir-builder-artifacts/emitpy ]; then
        touch concat_emitpy_results.json
        # Create renamed copies of ttnn files so emitpy can find them for comparison
        for file in ttir-builder-artifacts/emitpy/*; do
            if [ -f "$file" ] && [[ "$file" == *.py ]]; then
                # Get the basename of the .py file
                basename_file=$(basename "$file")

                # Create the corresponding TTNN filename by replacing "emitpy" with "ttnn" and ".py" with ".ttnn"
                ttnn_basename=$(echo "$basename_file" | sed 's/emitpy/ttnn/' | sed 's/\.py$/.ttnn/')
                ttnn_file="ttir-builder-artifacts/ttnn/$ttnn_basename"

                # Check if the corresponding TTNN file exists
                if [ -f "$ttnn_file" ]; then
                    echo "Found matching TTNN file: $ttnn_file"
                    FLATBUFFER=" --flatbuffer $ttnn_file "
                else
                    echo "No matching TTNN file found for $basename_file, running without flatbuffer comparison"
                    FLATBUFFER=""
                fi
                ttrt emitpy $TTRT_ARGS $FLATBUFFER $file
                jq -s add emitpy_results.json concat_emitpy_results.json > temp.json
                mv temp.json concat_emitpy_results.json
            fi
        done
        cp concat_emitpy_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_emitpy_${TEST_REPORT_PATH##*_} || true
    fi
    if [ -d ttir-builder-artifacts/emitc ]; then
        touch concat_emitc_results.json
        export TT_METAL_LIB="${INSTALL_DIR}/lib"
        ${INSTALL_DIR}/tools/ttnn-standalone/ci_compile_dylib.py --dir ttir-builder-artifacts/emitc
        # Create renamed copies of ttnn files so emitc can find them for comparison
        for file in ttir-builder-artifacts/emitc/*; do
            if [ -f "$file" ] && [[ "$file" == *.so ]]; then
                # Get the basename of the .so file
                basename_file=$(basename "$file")

                # Create the corresponding TTNN filename by replacing "emitc" with "ttnn" and ".so" with ".ttnn"
                ttnn_basename=$(echo "$basename_file" | sed 's/emitc/ttnn/' | sed 's/\.so$/.ttnn/')
                ttnn_file="ttir-builder-artifacts/ttnn/$ttnn_basename"

                # Check if the corresponding TTNN file exists
                if [ -f "$ttnn_file" ]; then
                    echo "Found matching TTNN file: $ttnn_file"
                    FLATBUFFER=" --flatbuffer $ttnn_file "
                else
                    echo "No matching TTNN file found for $basename_file, running without flatbuffer comparison"
                    FLATBUFFER=""
                fi
                ttrt emitc $TTRT_ARGS $FLATBUFFER $file
                jq -s add emitc_results.json concat_emitc_results.json > temp.json
                mv temp.json concat_emitc_results.json
            fi
        done
        cp concat_emitc_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_emitc_${TEST_REPORT_PATH##*_} || true
    fi
fi
