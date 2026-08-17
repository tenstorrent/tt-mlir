// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

// A training step holds one adamw op per parameter, and every one of them reads
// the same two bias-correction tensors. Each readback is a device-to-host sync,
// so the second op must reuse the first op's values instead of reading again.
module {
  func.func @adamw_two_params(%param0: tensor<1x1x64x64xf32>, %grad0: tensor<1x1x64x64xbf16>,
                              %exp_avg0: tensor<1x1x64x64xf32>, %exp_avg_sq0: tensor<1x1x64x64xf32>,
                              %param1: tensor<1x1x64x64xf32>, %grad1: tensor<1x1x64x64xbf16>,
                              %exp_avg1: tensor<1x1x64x64xf32>, %exp_avg_sq1: tensor<1x1x64x64xf32>,
                              %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>) {
    // CHECK: float [[LR:v[0-9]+]] = util_scalar_to_float(
    // CHECK: float [[BETA1:v[0-9]+]] = util_scalar_to_float(
    // CHECK: float [[BETA2:v[0-9]+]] = util_scalar_to_float(
    // CHECK: {{^ *}}ttml::metal::adamw({{.*}}, [[LR]], {{.*}}, [[BETA1]], [[BETA2]], {{.*}});
    %0:3 = "ttir.adamw"(%param0, %grad0, %exp_avg0, %exp_avg_sq0, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    // The second op reuses the values above: no further readback in between.
    // CHECK-NOT: util_scalar_to_float
    // CHECK: {{^ *}}ttml::metal::adamw({{.*}}, [[BETA1]], [[BETA2]], {{.*}});
    %1:3 = "ttir.adamw"(%param1, %grad1, %exp_avg1, %exp_avg_sq1, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32,
        beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32,
        weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0, %1#0 : tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>
  }
}
