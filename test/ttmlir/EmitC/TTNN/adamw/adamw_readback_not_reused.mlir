// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward --ttcore-wrap-device-module %s -o %t.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

// Companion to adamw_shared_bias_correction.mlir, which covers the reuse case.
// Here another device op consumes `lr` between the two adamw ops. After
// conversion every op is an opaque call, so the pattern cannot tell a reader
// from an in-place writer and reads `lr` again for the second op: one extra
// sync, never a stale value. `beta1_pow` and `beta2_pow` have no such consumer,
// so they are still read once.
//
// Written in the TTNN dialect so the op order is exactly as spelled out; going
// through the TTIR pipeline would leave the position of the `ttnn.abs` up to
// the pass that schedules it.

#dram = #ttnn.buffer_type<dram>
#param_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3),
                                  <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#scalar_layout = #ttnn.ttnn_layout<(d0) -> (0, d0),
                                   <1x1>, memref<1x1xf32, #dram>, <interleaved>>

func.func @adamw_lr_consumed_between(
    %param0: tensor<1x1x64x64xf32, #param_layout>, %grad0: tensor<1x1x64x64xf32, #param_layout>,
    %exp_avg0: tensor<1x1x64x64xf32, #param_layout>, %exp_avg_sq0: tensor<1x1x64x64xf32, #param_layout>,
    %param1: tensor<1x1x64x64xf32, #param_layout>, %grad1: tensor<1x1x64x64xf32, #param_layout>,
    %exp_avg1: tensor<1x1x64x64xf32, #param_layout>, %exp_avg_sq1: tensor<1x1x64x64xf32, #param_layout>,
    %lr: tensor<1xf32, #scalar_layout>, %beta1_pow: tensor<1xf32, #scalar_layout>,
    %beta2_pow: tensor<1xf32, #scalar_layout>) -> tensor<1xf32, #scalar_layout> {
  // CHECK: float [[LR:v[0-9]+]] = util_scalar_to_float(
  // CHECK: float [[BETA1:v[0-9]+]] = util_scalar_to_float(
  // CHECK: float [[BETA2:v[0-9]+]] = util_scalar_to_float(
  // CHECK: {{^ *}}ttml::metal::adamw({{.*}}, [[LR]], {{.*}}, [[BETA1]], [[BETA2]], {{.*}});
  "ttnn.adamw"(%param0, %grad0, %exp_avg0, %exp_avg_sq0, %lr, %beta1_pow, %beta2_pow) <{
      beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
      epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
      : (tensor<1x1x64x64xf32, #param_layout>, tensor<1x1x64x64xf32, #param_layout>,
         tensor<1x1x64x64xf32, #param_layout>, tensor<1x1x64x64xf32, #param_layout>,
         tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>) -> ()

  // A device op reading lr lands between the two adamw ops.
  // CHECK: ttnn::abs(
  %sched = "ttnn.abs"(%lr) : (tensor<1xf32, #scalar_layout>) -> tensor<1xf32, #scalar_layout>

  // lr is read again because the abs is now its last user; the two bias
  // correction terms keep the values read for the first op.
  // CHECK: float [[LR2:v[0-9]+]] = util_scalar_to_float(
  // CHECK: {{^ *}}ttml::metal::adamw({{.*}}, [[LR2]], {{.*}}, [[BETA1]], [[BETA2]], {{.*}});
  "ttnn.adamw"(%param1, %grad1, %exp_avg1, %exp_avg_sq1, %lr, %beta1_pow, %beta2_pow) <{
      beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
      epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
      : (tensor<1x1x64x64xf32, #param_layout>, tensor<1x1x64x64xf32, #param_layout>,
         tensor<1x1x64x64xf32, #param_layout>, tensor<1x1x64x64xf32, #param_layout>,
         tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>, tensor<1xf32, #scalar_layout>) -> ()
  return %sched : tensor<1xf32, #scalar_layout>
}
