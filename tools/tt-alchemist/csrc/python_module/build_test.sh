#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to build and run the dlopen test

set -e

echo "Building test_dlopen..."
g++ -std=c++20 -o test_dlopen test_dlopen.cpp -ldl

echo "Build successful!"
echo ""
echo "To run the test:"
echo "  ./test_dlopen <path-to-example_module.so>"
echo ""
echo "Example:"
echo "  ./test_dlopen ../../templates/python/local/example_module.cpython-311-x86_64-linux-gnu.so"

