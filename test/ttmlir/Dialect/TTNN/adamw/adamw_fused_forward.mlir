// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// The forward pass reads the parameter before the optimizer step updates it in
// place. The read must stay ahead of the step, and the parameter buffer must
// survive to the return.
module {
  func.func @fwd_then_step(%param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
                           %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>)
      -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) {
    // CHECK: %[[ACT:[0-9a-z_]+]] = "ttnn.abs"(%[[PARAM:[0-9a-z_]+]])
    %act = "ttir.abs"(%param) : (tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    // CHECK: "ttnn.adamw"(%[[PARAM]],
    // CHECK-SAME: -> ()
    %new:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    // CHECK-NOT: "ttnn.deallocate"(%[[PARAM]])
    // CHECK: return %[[PARAM]], %[[ACT]]
    return %new#0, %act : tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>
  }

  // CHECK-LABEL: func.func @fwd_add_then_step
  func.func @fwd_add_then_step(%param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
                               %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>)
      -> tensor<1x1x64x64xbf16> {
    // CHECK: %[[ACT:[0-9a-z_]+]] = "ttnn.abs"(%[[PARAM:[0-9a-z_]+]])
    %act = "ttir.abs"(%param) : (tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    // CHECK: "ttnn.adamw"(%[[PARAM]],
    %new:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    // CHECK: "ttnn.add"(%[[PARAM]], %[[ACT]])
    // CHECK-SAME: input_tensor_b_activations = []
    %out = "ttir.add"(%new#0, %act) : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>) -> tensor<1x1x64x64xbf16>
    return %out : tensor<1x1x64x64xbf16>
  }
}
