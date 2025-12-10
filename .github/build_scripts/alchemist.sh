#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e -o pipefail

# Build the tt-alchemist wheel for distribution (default target is editable wheel)
cmake --build $BUILD_DIR -- tt-alchemist-build-wheel-distribution

# Build C++ test executables for python_runner
cmake --build $BUILD_DIR -- test_python_runner test_python_runner_simple

# Copy test executables and required files to install directory
mkdir -p $INSTALL_DIR/tools/tt-alchemist/test
cp $BUILD_DIR/tools/tt-alchemist/csrc/test/test_python_runner $INSTALL_DIR/tools/tt-alchemist/test/
cp $BUILD_DIR/tools/tt-alchemist/csrc/test/test_python_runner_simple $INSTALL_DIR/tools/tt-alchemist/test/
cp $BUILD_DIR/tools/tt-alchemist/csrc/python_runner/libtt-alchemist-python-runner.so $INSTALL_DIR/tools/tt-alchemist/test/
cp $WORK_DIR/tools/tt-alchemist/test/simple_test_model.py $INSTALL_DIR/tools/tt-alchemist/test/
cp $WORK_DIR/tools/tt-alchemist/test/test_model.py $INSTALL_DIR/tools/tt-alchemist/test/

# Copy tt-metal dependencies needed by the test executables (for $ORIGIN RPATH)
TTMETAL_LIB_DIR=$WORK_DIR/third_party/tt-metal/src/tt-metal/build/lib
cp $TTMETAL_LIB_DIR/libtt_metal.so $INSTALL_DIR/tools/tt-alchemist/test/
cp $TTMETAL_LIB_DIR/_ttnncpp.so $INSTALL_DIR/tools/tt-alchemist/test/
cp $TTMETAL_LIB_DIR/libtt_stl.so $INSTALL_DIR/tools/tt-alchemist/test/
cp $TTMETAL_LIB_DIR/libfmt.so* $INSTALL_DIR/tools/tt-alchemist/test/

# upload artifact
echo "{\"name\":\"tt-alchemist-whl-$BUILD_NAME\",\"path\":\"$BUILD_DIR/tools/tt-alchemist/csrc/dist/tt_alchemist*.whl\"}," >> $UPLOAD_LIST
