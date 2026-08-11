// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// AdamW composite without max_exp_avg_sq (4 operands).
module {
  func.func @adamw(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                   %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                   %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // CHECK: "ttir.adamw"
    // CHECK-SAME: beta1 = 0.899999976 : f32
    // CHECK-SAME: lr = 1.000000e-03 : f32
    // CHECK-SAME: stochastic_rounding = false
    // CHECK-SAME: weight_decay = 0.00999999977 : f32
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    // The bias correction rides in as operands, never as attributes.
    // CHECK-NOT: beta1_pow
    // CHECK-NOT: stablehlo.composite
    %0:3 = stablehlo.composite "tenstorrent.adamw" %param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow {
      composite_attributes = {
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32
      },
      decomposition = @tenstorrent.adamw.impl
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
  func.func private @tenstorrent.adamw.impl(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>, %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    return %param, %exp_avg, %exp_avg_sq : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
}

// -----

// AdamW composite with max_exp_avg_sq (5 operands, amsgrad) and stochastic_rounding.
module {
  func.func @adamw_amsgrad(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                           %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                           %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>,
                           %max_exp_avg_sq: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.adamw"
    // CHECK-SAME: stochastic_rounding = true
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    // CHECK-NOT: stablehlo.composite
    %0:4 = stablehlo.composite "tenstorrent.adamw" %param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow, %max_exp_avg_sq {
      composite_attributes = {
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32,
        stochastic_rounding = true
      },
      decomposition = @tenstorrent.adamw.impl
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
  func.func private @tenstorrent.adamw.impl(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>, %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>, %max_exp_avg_sq: tensor<64x64xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    return %param, %exp_avg, %exp_avg_sq, %max_exp_avg_sq : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
}

// -----

// The bias-correction operands are normalized to f32, the same way the float
// attributes are: a frontend tracing a bf16 model hands them over in the
// model's own float width, and ttir.adamw only accepts f32.
module {
  func.func @adamw_bf16_bias_correction(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                        %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                        %beta1_pow: tensor<1xbf16>, %beta2_pow: tensor<1xbf16>)
      -> tensor<64x64xf32> {
    // CHECK-DAG: %[[BETA1:[0-9a-z_]+]] = "ttir.typecast"(%arg4{{.*}}) {{.*}} -> tensor<1xf32>
    // CHECK-DAG: %[[BETA2:[0-9a-z_]+]] = "ttir.typecast"(%arg5{{.*}}) {{.*}} -> tensor<1xf32>
    // CHECK: "ttir.adamw"
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    %0:3 = stablehlo.composite "tenstorrent.adamw" %param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow {
      composite_attributes = {
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32
      },
      decomposition = @tenstorrent.adamw.impl
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xbf16>, tensor<1xbf16>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
  func.func private @tenstorrent.adamw.impl(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>, %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %beta1_pow: tensor<1xbf16>, %beta2_pow: tensor<1xbf16>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    return %param, %exp_avg, %exp_avg_sq : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
}
