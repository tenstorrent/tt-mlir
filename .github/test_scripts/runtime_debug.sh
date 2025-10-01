#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Running TTNN Runtime Unit Tests"
$BUILD_DIR/runtime/test/common/gtest/test_generate_sys_desc
$BUILD_DIR/runtime/test/common/gtest/test_handle_float_bfloat_buffer_cast
$BUILD_DIR/runtime/test/common/gtest/test_handle_integer_buffer_cast
$BUILD_DIR/runtime/test/ttnn/gtest/test_runtime_worker
$BUILD_DIR/runtime/test/ttnn/gtest/test_tensor_serialization
