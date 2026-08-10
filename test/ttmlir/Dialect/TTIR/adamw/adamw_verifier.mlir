// RUN: ttmlir-opt -split-input-file -verify-diagnostics %s

// beta1_pow / beta2_pow are read back to host as plain floats, so they must hold
// exactly one f32 element.

module {
  func.func @beta1_pow_not_scalar(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                  %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                  %beta1_pow: tensor<4xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{beta1_pow must have exactly one element, got 4}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow) <{
        lr = 1.000000e-03 : f32, beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<4xf32>, tensor<1xf32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}

// -----

module {
  func.func @beta2_pow_wrong_dtype(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                   %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                   %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xbf16>)
      -> tensor<64x64xf32> {
    // expected-error @+1 {{beta2_pow must be f32, got 'bf16'}}
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow) <{
        lr = 1.000000e-03 : f32, beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xbf16>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}

// -----

// A rank-0 scalar is also accepted.
module {
  func.func @rank0_scalar_ok(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                             %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                             %beta1_pow: tensor<f32>, %beta2_pow: tensor<f32>)
      -> tensor<64x64xf32> {
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow) <{
        lr = 1.000000e-03 : f32, beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<f32>, tensor<f32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
}
