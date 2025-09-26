#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Set up PYTHONPATH to include the built packages
export PYTHONPATH="${{ steps.strings.outputs.build-output-dir }}/python_packages:${{ steps.strings.outputs.install-output-dir }}/tt-metal/ttnn:${{ steps.strings.outputs.install-output-dir }}/tt-metal:$PYTHONPATH"
mkdir -p third_party/tt-metal
mkdir -p third_party/tt-metal/src
ln -sf ${{ steps.strings.outputs.install-output-dir }}/tt-metal third_party/tt-metal/src/tt-metal

echo "Running PyKernel tests..."
pytest -v ${{ steps.strings.outputs.work-dir }}/test/pykernel/demo/test.py
pytest -v ${{ steps.strings.outputs.work-dir }}/test/ttnn-jit/test_unary_composite.py
rm -rf third_party/tt-metal
