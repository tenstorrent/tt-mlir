// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// residual tensor must have same shape as input
module attributes {} {
  func.func @rms_norm_pre_all_gather_input_shape(%input: tensor<1x1x128x64xbf16>, %residual: tensor<1x1x128x16xbf16>) -> tensor<1x1x128x32xbf16> {
    %0 = "ttnn.rms_norm_pre_all_gather"(%input, %residual) <{use_2d_core_grid = false}> : (tensor<1x1x128x64xbf16>, tensor<1x1x128x16xbf16>) -> tensor<1x1x128x32xbf16>
    return %0 : tensor<1x1x128x32xbf16>
  }
}
// CHECK: error: 'ttnn.rms_norm_pre_all_gather' op residual tensor shape must match the input tensor shape
