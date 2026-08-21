// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.adamw is deliberately unsupported. ttml does not
// expose the metal::adamw primitive through its Python bindings.

module {
  func.func @adamw(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                   %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>,
                   %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xf32>)
      -> tensor<1x1x64x64xf32> {
    // CHECK: failed to legalize operation 'ttnn.adamw'
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }
}
