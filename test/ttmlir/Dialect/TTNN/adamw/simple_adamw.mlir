// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // AdamW without AMSGrad (no max_exp_avg_sq).
  func.func @adamw(%param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
                   %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<1x1x64x64xbf16> {
    // CHECK: "ttnn.adamw"(%[[PARAM:[0-9a-z_]+]],
    // CHECK-SAME: -> ()
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    // CHECK-NOT: "ttnn.deallocate"(%[[PARAM]])
    // CHECK: return %[[PARAM]]
    return %0#0 : tensor<1x1x64x64xbf16>
  }
}
