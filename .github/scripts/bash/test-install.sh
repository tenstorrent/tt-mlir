#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Script to test a TTMLIR installation by building a standalone project against it.
# Usage: test-install.sh <install-prefix> [mlir-prefix]
#
# Arguments:
#   install-prefix: Path to the TTMLIR installation directory
#   mlir-prefix:    Optional MLIR installation prefix (default: /opt/ttmlir-toolchain)

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <install-prefix> [mlir-prefix]"
  echo "  install-prefix: Path to the TTMLIR installation directory"
  echo "  mlir-prefix:    Optional MLIR installation prefix (default: /opt/ttmlir-toolchain)"
  exit 1
fi

# Convert to absolute path
INSTALL_PREFIX="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"

MLIR_PREFIX="${2:-/opt/ttmlir-toolchain}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEST_DIR="${PROJECT_ROOT}/test/install"
BUILD_DIR="${PROJECT_ROOT}/build/test/install"

# Validate install prefix
if [ ! -d "${INSTALL_PREFIX}" ]; then
  echo "ERROR: Install prefix does not exist: ${INSTALL_PREFIX}"
  exit 1
fi

if [ ! -f "${INSTALL_PREFIX}/lib/cmake/ttmlir/TTMLIRConfig.cmake" ]; then
  echo "ERROR: TTMLIRConfig.cmake not found in ${INSTALL_PREFIX}/lib/cmake/ttmlir/"
  exit 1
fi

# Clean previous build
rm -rf "${BUILD_DIR}"

# Configure the test project
echo "Configuring test project..."
echo "  TTMLIR install: ${INSTALL_PREFIX}"
echo "  MLIR install:   ${MLIR_PREFIX}"
echo "  Build dir:      ${BUILD_DIR}"

cmake -B "${BUILD_DIR}" \
  -S "${TEST_DIR}" \
  -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}:${MLIR_PREFIX}" \
  -DTTMLIR_DIR="${INSTALL_PREFIX}" \
  -G Ninja

# Build the test project
echo "Building test project..."
cmake --build "${BUILD_DIR}"

# Verify the executable was created
if [ ! -f "${BUILD_DIR}/test_install" ]; then
  echo "ERROR: test_install executable not found in ${BUILD_DIR}"
  exit 1
fi

# Run the test
echo "Running test_install..."
"${BUILD_DIR}/test_install"

echo "SUCCESS: Install test passed!"
