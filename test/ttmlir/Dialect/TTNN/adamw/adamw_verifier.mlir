// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Mirrors test/ttmlir/Dialect/TTIR/adamw/adamw_verifier.mlir: the runtime reads
// beta1_pow / beta2_pow back to host as plain floats, so ttnn.adamw holds them
// to the same shape and dtype as ttir.adamw does.

#dram = #ttnn.buffer_type<dram>

#param_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                                  <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>,
                                  <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                   <1x1>, memref<1x1xf32, #dram>, <interleaved>>
#scalar_layout_4 = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                     <1x1>, memref<1x4xf32, #dram>, <interleaved>>
#scalar_layout_bf16 = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                        <1x1>, memref<1x1xbf16, #dram>, <interleaved>>

// Happy path: single-element f32 bias-correction operands.
module {
  func.func @adamw_ok(%param: tensor<64x64xf32, #param_layout>,
                      %grad: tensor<64x64xf32, #param_layout>,
                      %exp_avg: tensor<64x64xf32, #param_layout>,
                      %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                      %lr: tensor<1xf32, #scalar_layout>, %beta1_pow: tensor<1xf32, #scalar_layout>,
                      %beta2_pow: tensor<1xf32, #scalar_layout>) {
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>

#param_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                                  <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>,
                                  <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                   <1x1>, memref<1x1xf32, #dram>, <interleaved>>
#scalar_layout_4 = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                     <1x1>, memref<1x4xf32, #dram>, <interleaved>>

module {
  func.func @beta1_pow_not_scalar(%param: tensor<64x64xf32, #param_layout>,
                                  %grad: tensor<64x64xf32, #param_layout>,
                                  %exp_avg: tensor<64x64xf32, #param_layout>,
                                  %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                                  %lr: tensor<1xf32, #scalar_layout>, %beta1_pow: tensor<4xf32, #scalar_layout_4>,
                                  %beta2_pow: tensor<1xf32, #scalar_layout>) {
    // expected-error @+1 {{beta1_pow must have exactly one element, got 4}}
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<1xf32, #scalar_layout>, tensor<4xf32, #scalar_layout_4>, tensor<1xf32, #scalar_layout>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>

#param_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                                  <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>,
                                  <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                   <1x1>, memref<1x1xf32, #dram>, <interleaved>>
#scalar_layout_bf16 = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                        <1x1>, memref<1x1xbf16, #dram>, <interleaved>>

module {
  func.func @beta2_pow_wrong_dtype(%param: tensor<64x64xf32, #param_layout>,
                                   %grad: tensor<64x64xf32, #param_layout>,
                                   %exp_avg: tensor<64x64xf32, #param_layout>,
                                   %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                                   %lr: tensor<1xf32, #scalar_layout>, %beta1_pow: tensor<1xf32, #scalar_layout>,
                                   %beta2_pow: tensor<1xbf16, #scalar_layout_bf16>) {
    // expected-error @+1 {{beta2_pow must be f32, got 'bf16'}}
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>, tensor<1xbf16, #scalar_layout_bf16>) -> ()
    return
  }
}

// -----

#dram = #ttnn.buffer_type<dram>

#param_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                                  <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>,
                                  <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                   <1x1>, memref<1x1xf32, #dram>, <interleaved>>
#scalar_layout_rank0 = #ttnn.ttnn_layout<() -> (0, 0),
                                         <1x1>, memref<1x1xf32, #dram>, <interleaved>>

module {
  func.func @beta1_pow_rank0(%param: tensor<64x64xf32, #param_layout>,
                             %grad: tensor<64x64xf32, #param_layout>,
                             %exp_avg: tensor<64x64xf32, #param_layout>,
                             %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                             %lr: tensor<1xf32, #scalar_layout>, %beta1_pow: tensor<f32, #scalar_layout_rank0>,
                             %beta2_pow: tensor<1xf32, #scalar_layout>) {
    // expected-error @+1 {{beta1_pow must have rank of at least 1}}
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<1xf32, #scalar_layout>, tensor<f32, #scalar_layout_rank0>, tensor<1xf32, #scalar_layout>) -> ()
    return
  }
}
