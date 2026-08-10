// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  func.func @adamw(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                   %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>,
                   %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<1x1x64x64xf32> {
    // The bias-correction tensors are read back to floats before the call.
    // CHECK: float [[BETA1:v[0-9]+]] = util_scalar_to_float(
    // CHECK: float [[BETA2:v[0-9]+]] = util_scalar_to_float(
    // Without `max_exp_avg_sq` the fifth argument is `::std::nullopt`, and
    // epsilon must survive as 1e-8 rather than being rounded down to zero.
    // CHECK: {{^ *}}ttml::metal::adamw({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, ::std::nullopt, 0.00100000005, 0.899999976, 0.999000012, [[BETA1]], [[BETA2]], 9.99999993E-9, 0.00999999977, ::ttml::metal::StochasticRounding::Disabled);
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %beta1_pow, %beta2_pow) <{
        lr = 1.000000e-03 : f32,
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }
}
