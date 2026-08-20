// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  func.func @adamw(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                   %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>,
                   %step_size: tensor<1xf32>, %inv_sqrt_bc2: tensor<1xf32>, %decay_factor: tensor<1xf32>)
      -> tensor<1x1x64x64xf32> {
    // The step-varying scalars are handed to ttml as device tensors: no
    // util_scalar_to_float readback is emitted anywhere.
    // CHECK-NOT: util_scalar_to_float
    // Without `max_exp_avg_sq` the fifth argument is `::std::nullopt`, and
    // epsilon must survive as 1e-8 rather than being rounded down to zero.
    // CHECK: {{^ *}}ttml::metal::adamw_tensor_scalars({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, ::std::nullopt, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, 0.899999976f, 0.999000012f, 9.99999993E-9f, ::ttml::metal::StochasticRounding::Disabled);
    // CHECK-NOT: util_scalar_to_float
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %step_size, %inv_sqrt_bc2, %decay_factor) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }
}
