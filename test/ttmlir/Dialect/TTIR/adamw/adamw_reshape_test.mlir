// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

// ttml::metal::adamw only accepts 4D tensors, so non-4D operands are reshaped up
// to 4D and every result is reshaped back down.
module {
  // CHECK-LABEL: func.func @adamw_rank2
  func.func @adamw_rank2(%param: tensor<64x64xf32>, %grad: tensor<64x64xbf16>,
                         %exp_avg: tensor<64x64xf32>, %exp_avg_sq: tensor<64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) {
    // CHECK-DAG: "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]}>
    // CHECK-DAG: "ttir.reshape"(%arg3) <{shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]}>
    // The scalar operands are passed through untouched.
    // CHECK: "ttir.adamw"({{.*}}, %arg4, %arg5, %arg6)
    // CHECK-SAME: -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<64x64xf32>, tensor<64x64xbf16>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>)
    // Every result is reshaped back, so the updated moments reach the caller.
    // CHECK-COUNT-3: "ttir.reshape"({{.*}}) <{shape = [64 : i32, 64 : i32]}>
    return %0#0, %0#1, %0#2 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>
  }

  // A rank-4 op is already legal and must be left alone.
  // CHECK-LABEL: func.func @adamw_rank4
  func.func @adamw_rank4(%param: tensor<1x1x64x64xf32>, %grad: tensor<1x1x64x64xbf16>,
                         %exp_avg: tensor<1x1x64x64xf32>, %exp_avg_sq: tensor<1x1x64x64xf32>, %lr: tensor<1xf32>, %beta1_pow: tensor<1xf32>, %beta2_pow: tensor<1xf32>)
      -> tensor<1x1x64x64xf32> {
    // CHECK-NOT: "ttir.reshape"
    // CHECK: "ttir.adamw"
    %0:3 = "ttir.adamw"(%param, %grad, %exp_avg, %exp_avg_sq, %lr, %beta1_pow, %beta2_pow) <{
        beta1 = 0.899999976 : f32, beta2 = 0.999000012 : f32,
        epsilon = 1.000000e-08 : f32, weight_decay = 1.000000e-02 : f32}>
        : (tensor<1x1x64x64xf32>, tensor<1x1x64x64xbf16>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>)
          -> (tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>, tensor<1x1x64x64xf32>)
    return %0#0 : tensor<1x1x64x64xf32>
  }
}
