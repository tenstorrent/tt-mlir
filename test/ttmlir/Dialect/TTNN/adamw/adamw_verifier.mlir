// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Mirrors test/ttmlir/Dialect/TTIR/adamw/adamw_verifier.mlir: ttml consumes
// step_size / inv_sqrt_bc2 / decay_factor as single-element f32 device
// tensors, so ttnn.adamw holds them to the same shape and dtype as ttir.adamw
// does - one f32 element, rank >= 1.

#dram = #ttnn.buffer_type<dram>

#param_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1),
                                  <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>,
                                  <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                   <1x1>, memref<1x1xf32, #dram>, <interleaved>>

// Happy path: single-element f32 scalar operands.
module {
  func.func @adamw_ok(%param: tensor<64x64xf32, #param_layout>,
                      %grad: tensor<64x64xf32, #param_layout>,
                      %exp_avg: tensor<64x64xf32, #param_layout>,
                      %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                      %step_size: tensor<1xf32, #scalar_layout>, %inv_sqrt_bc2: tensor<1xf32, #scalar_layout>,
                      %decay_factor: tensor<1xf32, #scalar_layout>) {
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
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
  func.func @inv_sqrt_bc2_not_scalar(%param: tensor<64x64xf32, #param_layout>,
                                     %grad: tensor<64x64xf32, #param_layout>,
                                     %exp_avg: tensor<64x64xf32, #param_layout>,
                                     %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                                     %step_size: tensor<1xf32, #scalar_layout>, %inv_sqrt_bc2: tensor<4xf32, #scalar_layout_4>,
                                     %decay_factor: tensor<1xf32, #scalar_layout>) {
    // expected-error @+1 {{inv_sqrt_bc2 must have exactly one element, got 4}}
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
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

// Any width but f32 is rejected: ttml validates FLOAT32 for the scalar
// tensors. A bf16 model gets its scalars widened during composite
// legalization, so nothing bf16 reaches this op.
module {
  func.func @decay_factor_not_f32(%param: tensor<64x64xf32, #param_layout>,
                                  %grad: tensor<64x64xf32, #param_layout>,
                                  %exp_avg: tensor<64x64xf32, #param_layout>,
                                  %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                                  %step_size: tensor<1xf32, #scalar_layout>, %inv_sqrt_bc2: tensor<1xf32, #scalar_layout>,
                                  %decay_factor: tensor<1xbf16, #scalar_layout_bf16>) {
    // expected-error @+1 {{decay_factor must be f32, got 'bf16'}}
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
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
  func.func @inv_sqrt_bc2_rank0(%param: tensor<64x64xf32, #param_layout>,
                                %grad: tensor<64x64xf32, #param_layout>,
                                %exp_avg: tensor<64x64xf32, #param_layout>,
                                %exp_avg_sq: tensor<64x64xf32, #param_layout>,
                                %step_size: tensor<1xf32, #scalar_layout>, %inv_sqrt_bc2: tensor<f32, #scalar_layout_rank0>,
                                %decay_factor: tensor<1xf32, #scalar_layout>) {
    // expected-error @+1 {{inv_sqrt_bc2 must have rank of at least 1}}
    "ttnn.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<64x64xf32, #param_layout>, tensor<64x64xf32, #param_layout>,
           tensor<1xf32, #scalar_layout>, tensor<f32, #scalar_layout_rank0>, tensor<1xf32, #scalar_layout>) -> ()
    return
  }
}
