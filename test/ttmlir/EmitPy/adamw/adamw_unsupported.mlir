// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.adamw is deliberately unsupported. ttml does not
// expose the metal::adamw primitive through its Python bindings.

module {
  func.func @adamw(%param: tensor<64x64xf32>, %grad: tensor<64x64xbf16>,
                   %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>)
      -> tensor<64x64xf32> {
    // CHECK: failed to legalize operation 'ttnn.adamw'
    %0 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        beta1_pow = 0.899999976 : f32,
        beta2_pow = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xbf16>, tensor<64x64xf32>, tensor<64x64xf32>)
          -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}
