#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg $1: path to pytest test files
# arg $2: pytest marker expression to select tests to run
# arg $3: "run-ttrt" or predefined additional flags for pytest and ttrt

set -e -o pipefail

runttrt=""
TTRT_ARGS=""
PYTEST_ARGS=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
[[ "$RUNS_ON" == "p150" ]] && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"

for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH
if [[ "$runttrt" == "1" ]]; then
    ttrt run $TTRT_ARGS ttir-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
    cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_${TEST_REPORT_PATH##*_} || true
    ttrt run $TTRT_ARGS stablehlo-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
    if [ -d ttir-builder-artifacts/emitpy ]; then
        # Create renamed copies of ttnn files so emitpy can find them for comparison
        for file in ttir-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitpy/')
                cp "$file" "ttir-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitpy $TTRT_ARGS ttir-builder-artifacts/emitpy/ --flatbuffer ttir-builder-artifacts/ttnn/
        cp emitpy_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_emitpy_${TEST_REPORT_PATH##*_} || true
    fi
    if [ -d stablehlo-builder-artifacts/emitpy ]; then
        # Create renamed copies of ttnn files so emitpy can find them for comparison
        for file in stablehlo-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitpy/')
                cp "$file" "stablehlo-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitpy $TTRT_ARGS stablehlo-builder-artifacts/emitpy/ --flatbuffer stablehlo-builder-artifacts/ttnn/
        cp emitpy_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_stablehlo_emitpy_${TEST_REPORT_PATH##*_} || true
    fi
    if [ -d ttir-builder-artifacts/emitc ]; then
        find . -name _ttnncpp.so
        echo $TT_METAL_LIB
        export TT_METAL_LIB="${INSTALL_DIR}/lib"
        echo $TT_METAL_LIB
        ls ${TT_METAL_LIB}
        ${INSTALL_DIR}/tools/ttnn-standalone/ci_compile_dylib.py --dir ttir-builder-artifacts/emitc
        # Create renamed copies of ttnn files so emitc can find them for comparison
        for file in ttir-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitc/')
                cp "$file" "ttir-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitc $TTRT_ARGS ttir-builder-artifacts/emitc/ --flatbuffer ttir-builder-artifacts/ttnn/
        cp emitc_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
    fi
    if [ -d stablehlo-builder-artifacts/emitc ]; then
        export TT_METAL_LIB="${INSTALL_DIR}/lib"
        ${INSTALL_DIR}/tools/ttnn-standalone/ci_compile_dylib.py --dir stablehlo-builder-artifacts/emitc
        # Create renamed copies of ttnn files so emitc can find them for comparison
        for file in stablehlo-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitc/')
                cp "$file" "stablehlo-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitc $TTRT_ARGS stablehlo-builder-artifacts/emitc/ --flatbuffer stablehlo-builder-artifacts/ttnn/
        cp emitc_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
    fi
fi






#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# arg $1: path to pytest test files
# arg $2: pytest marker expression to select tests to run
# arg $3: "run-ttrt" or predefined additional flags for pytest and ttrt

set -e -o pipefail

runttrt=""
TTRT_ARGS=""
PYTEST_ARGS=""

[[ "$RUNS_ON" != "n150" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-exact-mesh"
[[ "$RUNS_ON" == "p150" ]] && TTRT_ARGS="$TTRT_ARGS --disable-eth-dispatch"

for flag in $3; do
    [[ "$flag" == "run-ttrt" ]] && runttrt=1
    [[ "$flag" == "require-opmodel" ]] && PYTEST_ARGS="$PYTEST_ARGS --require-opmodel"
done

pytest "$1" -m "$2" $PYTEST_ARGS -v --junit-xml=$TEST_REPORT_PATH
if [[ "$runttrt" == "1" ]]; then
    ttrt run $TTRT_ARGS ttir-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
    cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_${TEST_REPORT_PATH##*_} || true
    ttrt run $TTRT_ARGS stablehlo-builder-artifacts/
    cp run_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
    if [ -d ttir-builder-artifacts/emitc ]; then
        find . -name _ttnncpp.so
        echo $TT_METAL_LIB
        export TT_METAL_LIB="${INSTALL_DIR}/lib"
        echo $TT_METAL_LIB
        ls ${TT_METAL_LIB}/
        ${INSTALL_DIR}/tools/ttnn-standalone/ci_compile_dylib.py --dir ttir-builder-artifacts/emitc
        # Create renamed copies of ttnn files so emitc can find them for comparison
        for file in ttir-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitc/')
                cp "$file" "ttir-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitc $TTRT_ARGS ttir-builder-artifacts/emitc/ --flatbuffer ttir-builder-artifacts/ttnn/
        cp emitc_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
    fi
    if [ -d ttir-builder-artifacts/emitpy ]; then
        # Create renamed copies of ttnn files so emitpy can find them for comparison
        for file in ttir-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitpy/')
                cp "$file" "ttir-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitpy $TTRT_ARGS ttir-builder-artifacts/emitpy/ --flatbuffer ttir-builder-artifacts/ttnn/
        cp emitpy_results.json ${TTRT_REPORT_PATH%_*}_ttir_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_ttir_emitpy_${TEST_REPORT_PATH##*_} || true
    fi
    if [ -d stablehlo-builder-artifacts/emitpy ]; then
        # Create renamed copies of ttnn files so emitpy can find them for comparison
        for file in stablehlo-builder-artifacts/ttnn/*; do
            if [ -f "$file" ]; then
                basename_file=$(basename "$file")
                renamed_file=$(echo "$basename_file" | sed 's/ttnn/emitpy/')
                cp "$file" "stablehlo-builder-artifacts/ttnn/$renamed_file"
            fi
        done
        ttrt emitpy $TTRT_ARGS stablehlo-builder-artifacts/emitpy/ --flatbuffer stablehlo-builder-artifacts/ttnn/
        cp emitpy_results.json ${TTRT_REPORT_PATH%_*}_stablehlo_${TTRT_REPORT_PATH##*_} || true
        cp ttrt_report.xml ${TEST_REPORT_PATH%_*}_stablehlo_emitpy_${TEST_REPORT_PATH##*_} || true
    fi
fi
