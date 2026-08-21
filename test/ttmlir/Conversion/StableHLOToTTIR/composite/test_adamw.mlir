// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// AdamW composite without max_exp_avg_sq (7 operands). The composite carries
// lr / beta1_pow / beta2_pow; the legalization derives the three scalars
// ttml::metal::adamw_tensor_scalars consumes, on device:
//   step_size    = lr / (1 - beta1_pow)
//   inv_sqrt_bc2 = 1 / sqrt(1 - beta2_pow)
//   decay_factor = 1 - lr * weight_decay
module {
  func.func @adamw(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                   %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                   %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<64x64xf32> {
    // step_size = lr / (1 - beta1_pow)
    // CHECK: %[[ONE1:[0-9]+]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1>}>
    // CHECK: %[[BC1:[0-9]+]] = "ttir.subtract"(%[[ONE1]], %arg5)
    // CHECK: %[[STEP_SIZE:[0-9]+]] = "ttir.div"(%arg4, %[[BC1]])
    // inv_sqrt_bc2 = rsqrt(1 - beta2_pow)
    // CHECK: %[[ONE2:[0-9]+]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1>}>
    // CHECK: %[[BC2:[0-9]+]] = "ttir.subtract"(%[[ONE2]], %arg6)
    // CHECK: %[[INV_SQRT_BC2:[0-9]+]] = "ttir.rsqrt"(%[[BC2]])
    // decay_factor = 1 - lr * weight_decay
    // CHECK: %[[WD:[0-9]+]] = "ttir.full"() <{fill_value = 0.00999999977 : f32, shape = array<i32: 1>}>
    // CHECK: %[[LR_WD:[0-9]+]] = "ttir.multiply"(%arg4, %[[WD]])
    // CHECK: %[[ONE3:[0-9]+]] = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 1>}>
    // CHECK: %[[DECAY_FACTOR:[0-9]+]] = "ttir.subtract"(%[[ONE3]], %[[LR_WD]])
    // CHECK: "ttir.adamw"(%arg0, %arg1, %arg2, %arg3, %[[STEP_SIZE]], %[[INV_SQRT_BC2]], %[[DECAY_FACTOR]])
    // lr and weight_decay never survive as attributes: the attribute dictionary
    // is matched in full to say so.
    // CHECK-SAME: <{beta1 = 0.899999976 : f32, beta2 = 9.990000e-01 : f32, epsilon = 9.99999993E-9 : f32, stochastic_rounding = false}>
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
                           %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                           %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>,
                           %max_exp_avg_sq: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.adamw"
    // CHECK-SAME: <{beta1 = 0.899999976 : f32, beta2 = 9.990000e-01 : f32, epsilon = 9.99999993E-9 : f32, stochastic_rounding = true}>
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

// -----

// A frontend tracing a bf16 model hands the scalar operands over in the model's
// own float width. ttml validates FLOAT32 for the scalar tensors, so the
// legalization widens them once before deriving the step quantities; the
// derived arithmetic then runs in f32, which the bias correction needs anyway
// (1 - beta2^t underflows badly in bf16).
module {
  func.func @adamw_bf16_scalars(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>,
                                %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>,
                                %lr: tensor<1xbf16>, %beta1_pow: tensor<1xbf16>, %beta2_pow: tensor<1xbf16>)
      -> tensor<64x64xf32> {
    // CHECK-COUNT-3: "ttir.typecast"({{%arg[4-6]}}) {{.*}} -> tensor<1xf32>
    // CHECK: "ttir.adamw"
    // CHECK-SAME: beta1 = 0.899999976 : f32
    // CHECK-SAME: (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    %0:3 = stablehlo.composite "tenstorrent.adamw" %param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow {
      composite_attributes = {
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32
      },
      decomposition = @tenstorrent.adamw.impl
    } : (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    return %0#0 : tensor<64x64xf32>
  }
  func.func private @tenstorrent.adamw.impl(%param: tensor<64x64xf32>, %grad: tensor<64x64xf32>, %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %lr: tensor<1xbf16>, %beta1_pow: tensor<1xbf16>, %beta2_pow: tensor<1xbf16>) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    return %param, %exp_avg, %exp_avg_sq : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }
}
