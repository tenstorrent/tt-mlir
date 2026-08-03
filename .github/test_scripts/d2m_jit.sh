#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

export PYTHONPATH="$BUILD_DIR/python_packages:$WORK_DIR/test/d2m-jit:$INSTALL_DIR/tt-metal/ttnn:$INSTALL_DIR/tt-metal:$PYTHONPATH"

# The in-process runtime expects tt-metal at third_party/tt-metal/src/tt-metal.
mkdir -p $WORK_DIR/third_party/tt-metal/src
ln -sf $INSTALL_DIR/tt-metal $WORK_DIR/third_party/tt-metal/src/tt-metal

# lit and pytest both emit junit xml; keep them in separate report files so one
# does not clobber the other (both still match report_*.xml for collection).
LIT_REPORT_PATH="${TEST_REPORT_PATH%.xml}_lit.xml"

SIM_REPORT_PATH="${TEST_REPORT_PATH%.xml}_sim.xml"

echo "Running d2m-jit tests (RUNS_ON=$RUNS_ON)..."
# Full suite: FileCheck lit tests + every pytest module. Runs on every PR.
# Pass the directory, not a test_*.py glob: the glob only matches the top level
# and would silently skip subdirectories such as test/d2m-jit/sim/.
llvm-lit -v --xunit-xml-output "$LIT_REPORT_PATH" "$BUILD_DIR/test/d2m-jit/lit"
pytest -v "$WORK_DIR"/test/d2m-jit --junit-xml="$TEST_REPORT_PATH"

# Re-run the same kernels on the pure-Python/torch simulator backend. Every test
# carries its own torch golden, so this checks the simulator against the same
# reference the device run uses -- no hand-copied sim suite, and no separate
# device-vs-sim comparison needed. Tests marked `device_only` skip themselves
# here (see conftest.py).
echo "Re-running d2m-jit tests on the simulator backend..."
D2M_JIT_BACKEND=sim pytest -v "$WORK_DIR"/test/d2m-jit \
    --junit-xml="$SIM_REPORT_PATH"

# cleanup
rm -rf $WORK_DIR/third_party/tt-metal
