#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Script to install and test TTMLIR by building a standalone project
# against the installation using a standalone test project in `test/install`.
#
# This script expects the following environment variables to be set:
#   WORK_DIR    - Repository root directory
#   BUILD_DIR   - Build artifacts directory (where TTMLIR was built)
#   BUILD_NAME  - Name of the build image (for informational purposes)
#
# The script will install TTMLIR from BUILD_DIR to TTMLIR_INSTALL_DIR, then test it.

set -euo pipefail

if [ -z "${WORK_DIR:-}" ] || [ ! -d "${WORK_DIR}" ]; then
  echo "ERROR: WORK_DIR environment variable is not set or directory does not exist"
  exit 1
fi

if [ -z "${BUILD_DIR:-}" ] || [ ! -d "${BUILD_DIR}" ]; then
  echo "ERROR: BUILD_DIR environment variable is not set or directory does not exist"
  exit 1
fi

MLIR_PREFIX="${TTMLIR_TOOLCHAIN_DIR:-/opt/ttmlir-toolchain}"
TTMLIR_INSTALL_DIR="${BUILD_DIR}/ttmlir-install"
TEST_DIR="${WORK_DIR}/test/install"
TEST_BUILD_DIR="${WORK_DIR}/build/test/install"

# Install TTMLIR
echo "Installing TTMLIR..."
echo "  Build dir:      ${BUILD_DIR}"
echo "  Install dir:    ${TTMLIR_INSTALL_DIR}"
rm -rf "${TTMLIR_INSTALL_DIR}"
cmake --install "${BUILD_DIR}" --prefix "${TTMLIR_INSTALL_DIR}"

if [ ! -f "${TTMLIR_INSTALL_DIR}/lib/cmake/ttmlir/TTMLIRConfig.cmake" ]; then
  echo "ERROR: Install failed. TTMLIRConfig.cmake not found in ${TTMLIR_INSTALL_DIR}/lib/cmake/ttmlir/"
  exit 1
fi

# Configure, build, and run the test project
rm -rf "${TEST_BUILD_DIR}"
echo "Configuring test project..."
echo "  Build name:     ${BUILD_NAME:-unknown}"
echo "  TTMLIR install: ${TTMLIR_INSTALL_DIR}"
echo "  MLIR install:   ${MLIR_PREFIX}"
echo "  Test build dir: ${TEST_BUILD_DIR}"

cmake -B "${TEST_BUILD_DIR}" \
  -S "${TEST_DIR}" \
  -DCMAKE_PREFIX_PATH="${TTMLIR_INSTALL_DIR}:${MLIR_PREFIX}" \
  -DTTMLIR_DIR="${TTMLIR_INSTALL_DIR}" \
  -G Ninja

echo "Building test project..."
cmake --build "${TEST_BUILD_DIR}"

if [ ! -f "${TEST_BUILD_DIR}/test_install" ]; then
  echo "ERROR: test_install executable not found in ${TEST_BUILD_DIR}"
  exit 1
fi

echo "Running test_install..."
"${TEST_BUILD_DIR}/test_install"

echo "SUCCESS: Install test passed!"
