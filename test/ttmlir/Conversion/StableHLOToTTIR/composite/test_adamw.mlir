// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// AdamW composite without max_exp_avg_sq (7 operands).
module {
  func.func @adamw(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                   %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // CHECK: "ttir.adamw"
    // CHECK-SAME: beta1 = 0.899999976 : f32
    // CHECK-SAME: stochastic_rounding = false
    // CHECK-SAME: weight_decay = 0.00999999977 : f32
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK-NOT: stablehlo.composite
    %0:3 = stablehlo.composite "tenstorrent.adamw" %param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow {
      composite_attributes = {
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32
      },
      decomposition = @tenstorrent.adamw.impl
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
  func.func private @tenstorrent.adamw.impl(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>, %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    return %param, %exp_avg, %exp_avg_sq : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
}

// -----

// AdamW composite with max_exp_avg_sq (8 operands, amsgrad) and stochastic_rounding.
module {
  func.func @adamw_amsgrad(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                           %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>,
                           %max_exp_avg_sq: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.adamw"
    // CHECK-SAME: stochastic_rounding = true
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK-NOT: stablehlo.composite
    %0:4 = stablehlo.composite "tenstorrent.adamw" %param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow, %max_exp_avg_sq {
      composite_attributes = {
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32,
        stochastic_rounding = true
      },
      decomposition = @tenstorrent.adamw.impl
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
  func.func private @tenstorrent.adamw.impl(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>, %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>, %max_exp_avg_sq: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    return %param, %exp_avg, %exp_avg_sq, %max_exp_avg_sq : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
}
