// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

// A bf16 model hands lr and the bias-correction terms over in bf16. They reach
// the backend as bf16 tensors, with no typecast inserted anywhere on the way:
// the readback converts from any float width, so paying for a device op to
// widen a single element would be wasted work.
module {
  func.func @adamw_bf16_scalars(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                                %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>,
                                %lr: tensor<1xbf16>, %beta1_pow: tensor<1xbf16>, %beta2_pow: tensor<1xbf16>)
      -> tensor<1x1x64x64xf32> {
    // CHECK-NOT: ttnn::typecast
    // CHECK: float [[LR:v[0-9]+]] = util_scalar_to_float(
    // CHECK: float [[BETA1:v[0-9]+]] = util_scalar_to_float(
    // CHECK: float [[BETA2:v[0-9]+]] = util_scalar_to_float(
    // CHECK: {{^ *}}ttml::metal::adamw({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, ::std::nullopt, [[LR]], 0.899999976f, 0.999000012f, [[BETA1]], [[BETA2]], 9.99999993E-9f, 0.00999999977f, ::ttml::metal::StochasticRounding::Disabled);
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xbf16>, tensor<1xbf16>, tensor<1xbf16>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }
}
