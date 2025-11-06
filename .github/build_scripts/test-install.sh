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
#   INSTALL_DIR - Install artifacts directory (where TTMLIR will be installed)
#   BUILD_NAME  - Name of the build image (for informational purposes)
#
# The script will install TTMLIR from BUILD_DIR to INSTALL_DIR, then test it.

set -euo pipefail

# Validate required environment variables
if [ -z "${WORK_DIR:-}" ]; then
  echo "ERROR: WORK_DIR environment variable is not set"
  exit 1
fi

if [ -z "${BUILD_DIR:-}" ]; then
  echo "ERROR: BUILD_DIR environment variable is not set"
  exit 1
fi

if [ -z "${INSTALL_DIR:-}" ]; then
  echo "ERROR: INSTALL_DIR environment variable is not set"
  exit 1
fi

# Use default MLIR prefix if not set
MLIR_PREFIX="${MLIR_PREFIX:-/opt/ttmlir-toolchain}"

# Set up paths
PROJECT_ROOT="${WORK_DIR}"
TEST_DIR="${PROJECT_ROOT}/test/install"
TEST_BUILD_DIR="${PROJECT_ROOT}/build/test/install"

# Validate build directory
if [ ! -d "${BUILD_DIR}" ]; then
  echo "ERROR: Build directory does not exist: ${BUILD_DIR}"
  exit 1
fi

# Install TTMLIR to the install directory
echo "Installing TTMLIR..."
echo "  Build dir:      ${BUILD_DIR}"
echo "  Install dir:    ${INSTALL_DIR}"
rm -rf "${INSTALL_DIR}"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}"

# Validate installation
if [ ! -d "${INSTALL_DIR}" ]; then
  echo "ERROR: Install directory was not created: ${INSTALL_DIR}"
  exit 1
fi

if [ ! -f "${INSTALL_DIR}/lib/cmake/ttmlir/TTMLIRConfig.cmake" ]; then
  echo "ERROR: Install failed. TTMLIRConfig.cmake not found in ${INSTALL_DIR}/lib/cmake/ttmlir/"
  exit 1
fi

# Clean previous build of the test project
rm -rf "${TEST_BUILD_DIR}"

# Configure the test project
echo "Configuring test project..."
echo "  Build name:     ${BUILD_NAME:-unknown}"
echo "  TTMLIR install: ${INSTALL_DIR}"
echo "  MLIR install:   ${MLIR_PREFIX}"
echo "  Test build dir: ${TEST_BUILD_DIR}"

cmake -B "${TEST_BUILD_DIR}" \
  -S "${TEST_DIR}" \
  -DCMAKE_PREFIX_PATH="${INSTALL_DIR}:${MLIR_PREFIX}" \
  -DTTMLIR_DIR="${INSTALL_DIR}" \
  -G Ninja

# Build the test project
echo "Building test project..."
cmake --build "${TEST_BUILD_DIR}"

# Verify the executable was created
if [ ! -f "${TEST_BUILD_DIR}/test_install" ]; then
  echo "ERROR: test_install executable not found in ${TEST_BUILD_DIR}"
  exit 1
fi

# Run the test
echo "Running test_install..."
"${TEST_BUILD_DIR}/test_install"

echo "SUCCESS: Install test passed!"
