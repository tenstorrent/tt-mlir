// RUN: ttmlir-opt -split-input-file -verify-diagnostics %s

// step_size / inv_sqrt_bc2 / decay_factor go to ttml as single-element f32
// device tensors (ttml::metal::adamw_tensor_scalars validates FLOAT32), so
// each must hold exactly one f32 element.

module {
  func.func @inv_sqrt_bc2_not_scalar(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                     %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                     %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<4xf32>, %decay_factor: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{inv_sqrt_bc2 must have exactly one element, got 4}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<4xf32>, tensor<1xf32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}

// -----

// A non-f32 scalar is rejected: ttml validates FLOAT32 for the scalar tensors.
module {
  func.func @decay_factor_not_f32(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                  %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                  %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xi32>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{decay_factor must be f32, got 'i32'}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xi32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}

// -----

// bf16 scalars are rejected too: any width but f32 is. A bf16 model gets its
// scalars widened once during composite legalization, before the derived
// arithmetic, so nothing bf16 ever reaches this op.
module {
  func.func @step_size_bf16(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                            %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                            %step_size: tensor<1xbf16>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{step_size must be f32, got 'bf16'}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xbf16>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}

// -----

// Rank 0 is rejected: it does not survive the TTNN layout path, so it is caught
// here rather than in the backend.
module {
  func.func @rank0_scalar_rejected(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                   %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                   %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<f32>, %decay_factor: tensor<f32>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{inv_sqrt_bc2 must have rank of at least 1}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<f32>, tensor<f32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}

// -----

// A dynamic shape cannot be checked for its element count, so it is rejected
// outright rather than tripping the static-shape assert inside
// getNumElements().
module {
  func.func @step_size_dynamic_shape(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                     %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                     %step_size: tensor<?xf32>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{step_size must have a static shape}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{ beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<?xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}
