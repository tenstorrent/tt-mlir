// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // AdamW without AMSGrad (no max_exp_avg_sq). The step-varying scalars are
  // single-element tensor operands, so the step's bias correction never enters
  // the IR and the same program is reused every step.
  func.func @adamw(%param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
                   %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>,
                   %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xf32>)
      -> tensor<1x1x64x64xbf16> {
    // CHECK: "ttnn.adamw"(%[[PARAM:[0-9a-z_]+]],
    // The attribute dictionary is matched in full: the step-varying scalars
    // are operands now, so the only way to assert they are not attributes is to
    // pin down everything that is one. A bare CHECK-NOT would be vacuous here -
    // it only searches after the last match, and attributes print before the
    // operand types.
    // CHECK-SAME: <{beta1 = 0.899999976 : f32, beta2 = 9.990000e-01 : f32, epsilon = 9.99999993E-9 : f32, stochastic_rounding = false}>
    // CHECK-SAME: -> ()
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    // CHECK-NOT: "ttnn.deallocate"(%[[PARAM]])
    // CHECK: return %[[PARAM]]
    return %0#0 : tensor<1x1x64x64xbf16>
  }
}
