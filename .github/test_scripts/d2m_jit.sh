#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

# Unlike ttnn-jit, d2m-jit ships as a plain python package under
# build/python_packages (built by the `d2m-jit` CMake target on the tracy
# flavor), so there is no wheel to download/install. env/activate already puts
# build/python_packages on PYTHONPATH; set it explicitly anyway, and add the
# tt-metal python paths plus test/d2m-jit so the suite's `runner`/`utils`
# imports resolve under pytest.
export PYTHONPATH="$BUILD_DIR/python_packages:$WORK_DIR/test/d2m-jit:$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal:$PYTHONPATH"

# The in-process runtime expects tt-metal at third_party/tt-metal/src/tt-metal.
mkdir -p $WORK_DIR/third_party/tt-metal/src
ln -sf $INSTALL_DIR/tt-metal $WORK_DIR/third_party/tt-metal/src/tt-metal

# lit and pytest both emit junit xml; keep them in separate report files so one
# does not clobber the other (both still match report_*.xml for collection).
LIT_REPORT_PATH="${TEST_REPORT_PATH%.xml}_lit.xml"

echo "Running d2m-jit tests (RUNS_ON=$RUNS_ON)..."
if [ "$1" == "nightly" ]; then
    # Full suite: FileCheck lit tests + every pytest module. Runs every nightly
    # and on PRs that touch test/d2m-jit or tools/d2m-jit (d2m-jit-nightly
    # component in .github/settings/optional-components.yml).
    llvm-lit -v --xunit-xml-output "$LIT_REPORT_PATH" "$BUILD_DIR/test/d2m-jit/lit"
    pytest -v "$WORK_DIR"/test/d2m-jit/test_*.py --junit-xml="$TEST_REPORT_PATH"
else
    # Always-on PR smoke: a single, device-generic eltwise kernel that exercises
    # the build + in-process execution path on every runner.
    pytest -v "$WORK_DIR/test/d2m-jit/test_simple.py" --junit-xml="$TEST_REPORT_PATH"
fi

# cleanup
rm -rf $WORK_DIR/third_party/tt-metal
