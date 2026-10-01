// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  func.func @adamw(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                   %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<1x1x64x64xf32> {
    // lr / beta*_pow are read back from their device tensors.
    // CHECK-LABEL: adamw
    // CHECK-COUNT-3: ::ttnn::getScalarFromTensor<float>(
    // CHECK: {{^ *}}ttml::metal::adamw(
    // CHECK-SAME: ::std::nullopt
    // CHECK-SAME: ::ttml::metal::StochasticRounding::Disabled
    // CHECK-NOT: ::std::mt19937
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }

  // TTML requires a stochastic rounding seed iff stochastic rounding is
  // enabled, so the trailing argument is emitted only in this variant.
  func.func @adamw_stochastic_rounding(
      %param: tensor<1x1x64x64xbf16>, %grad: tensor<1x1x64x64xbf16>,
      %exp_avg: tensor<1x1x64x64xbf16>, %exp_avg_sq: tensor<1x1x64x64xbf16>,
      %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<1x1x64x64xbf16> {
    // CHECK-LABEL: adamw_stochastic_rounding
    // CHECK: {{^ *}}ttml::metal::adamw(
    // CHECK-SAME: ::ttml::metal::StochasticRounding::Enabled
    // CHECK-SAME: static thread_local ::std::mt19937
    // CHECK-SAME: static_cast<uint32_t>(generator())
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32,
        stochastic_rounding = true}>
        : (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xbf16>)
    return %0#0 : tensor<1x1x64x64xbf16>
  }
}
