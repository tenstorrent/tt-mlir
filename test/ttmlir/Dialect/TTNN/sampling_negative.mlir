// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// The ttnn::sampling kernel only supports between 1 and 32 users (one core
// per user). Verify that out-of-range batch sizes are rejected at verifier
// time instead of faulting deeper inside the kernel.
// tt-metal kernel constraint: https://github.com/tenstorrent/tt-metal/issues/47522

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_vals  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_idx   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_param = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_p     = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out   = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x2xsi32, #dram>, <interleaved>>

module {
  func.func @sampling_batch_64_rejected(
      %arg0: tensor<64x64xbf16, #ttnn_layout_vals>,
      %arg1: tensor<64x64xsi32, #ttnn_layout_idx>,
      %arg2: tensor<64xui32, #ttnn_layout_param>,
      %arg3: tensor<64xbf16, #ttnn_layout_p>,
      %arg4: tensor<64xbf16, #ttnn_layout_p>)
      -> tensor<64xsi32, #ttnn_layout_out> {
    // CHECK: error: 'ttnn.sampling' op batch (64) must be in [1, 32] (kernel limit)
    %0 = "ttnn.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<64x64xbf16, #ttnn_layout_vals>,
           tensor<64x64xsi32, #ttnn_layout_idx>,
           tensor<64xui32, #ttnn_layout_param>,
           tensor<64xbf16, #ttnn_layout_p>,
           tensor<64xbf16, #ttnn_layout_p>)
        -> tensor<64xsi32, #ttnn_layout_out>
    return %0 : tensor<64xsi32, #ttnn_layout_out>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_vals0  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_idx0   = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

// batch == 0 (empty) — also out of range, must be rejected.
module {
  func.func @sampling_batch_0_rejected(
      %arg0: tensor<0x64xbf16, #ttnn_layout_vals0>,
      %arg1: tensor<0x64xsi32, #ttnn_layout_idx0>,
      %arg2: tensor<0xui32>,
      %arg3: tensor<0xbf16>,
      %arg4: tensor<0xbf16>)
      -> tensor<0xsi32> {
    // CHECK: error: 'ttnn.sampling' op batch (0) must be in [1, 32] (kernel limit)
    %0 = "ttnn.sampling"(%arg0, %arg1, %arg2, %arg3, %arg4)
        : (tensor<0x64xbf16, #ttnn_layout_vals0>,
           tensor<0x64xsi32, #ttnn_layout_idx0>,
           tensor<0xui32>,
           tensor<0xbf16>,
           tensor<0xbf16>)
        -> tensor<0xsi32>
    return %0 : tensor<0xsi32>
  }
}
