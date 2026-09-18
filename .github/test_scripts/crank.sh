#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# tt-crank tests: the C++ unit tests (gtest) followed by the Python suite (pytest).
#
# arg $1: "sim"    - route the runtime through ttsim (builder lane, no device)
#         "device" - run on the attached device
#
# Needs a tracy build with TTMLIR_ENABLE_CRANK=ON. The build-dir artifact carries
# build/tt-crank (test binary, libs, python extension) and build/ttsim_home (the
# simulator staging dir baked into libtt_crank.so). Benchmarks are excluded by
# tt-crank/pytest.ini (norecursedirs). $REQUIREMENTS, if set, is pip-installed
# first (same as pytest.sh).

set -e -o pipefail

case "${1:-}" in
    sim)    SIM_ARG="sim"; PYTEST_ARGS="--sim" ;;
    device) SIM_ARG="";    PYTEST_ARGS="" ;;
    *)      echo "usage: crank.sh sim|device" >&2; exit 1 ;;
esac

if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi

CRANK_PY="$WORK_DIR/tt-crank/python"
TT_METAL_LINK="$WORK_DIR/third_party/tt-metal/src/tt-metal"

# Leave the checkout as we found it, also on failure (the job runs several
# tests in one checkout).
cleanup() {
    rm -f "$CRANK_PY"/tt_crank/torch/_native*.so "$CRANK_PY/tracy/_original" "$CRANK_PY/ttnn/_original"
    rm -rf "$WORK_DIR/third_party/tt-metal"
}
trap cleanup EXIT

# tt-crank bakes third_party/tt-metal/src/tt-metal as TT_METAL_HOME/RUNTIME_ROOT
# (and the test job exports TT_METAL_HOME pointing there): alias it to the
# install tree, same as pykernel.sh / d2m_jit.sh do.
mkdir -p "$(dirname "$TT_METAL_LINK")"
ln -sfn "$INSTALL_DIR/tt-metal" "$TT_METAL_LINK"

# What `tt-crank/scripts/install-py` (editable install) would wire up: the
# extension next to its python package, and tt-metal's tracy/ttnn packages
# behind the re-export wrappers.
ln -sfn "$BUILD_DIR"/tt-crank/src/torch/_native*.so "$CRANK_PY/tt_crank/torch/"
ln -sfn "$INSTALL_DIR/tt-metal/tools/tracy" "$CRANK_PY/tracy/_original"
ln -sfn "$INSTALL_DIR/tt-metal/ttnn/ttnn" "$CRANK_PY/ttnn/_original"
export PYTHONPATH="$CRANK_PY${PYTHONPATH:+:$PYTHONPATH}"

# libtt_crank / libtt_crank_common are found via RUNPATH (absolute build path,
# identical across jobs); list them explicitly as well so a workspace path
# mismatch fails at the tt-mlir libs, not here.
export LD_LIBRARY_PATH="$BUILD_DIR/tt-crank/src:$BUILD_DIR/tt-crank/src/common${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Running tt-crank C++ unit tests ($1)"
"$WORK_DIR/tt-crank/scripts/test" $SIM_ARG -- --gtest_output="xml:${TEST_REPORT_PATH%_*}_gtest_${TEST_REPORT_PATH##*_}"

echo "Running tt-crank Python tests ($1)"
cd "$WORK_DIR/tt-crank"
pytest -v tests/python $PYTEST_ARGS --junit-xml="$TEST_REPORT_PATH"
