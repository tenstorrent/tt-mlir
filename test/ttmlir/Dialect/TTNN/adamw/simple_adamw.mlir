// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // AdamW without AMSGrad (no max_exp_avg_sq).
  func.func @adamw(%param: tensor<64x64xbf16>, %grad: tensor<64x64xbf16>,
                   %exp_avg: tensor<64x64xbf16>, %exp_avg_sq: tensor<64x64xbf16>)
      -> tensor<64x64xbf16> {
    // CHECK: "ttnn.adamw"
    %0 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>)
          -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}
