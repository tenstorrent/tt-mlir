// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --convert-ttir-to-ttnn --ttnn-workaround --canonicalize %s | FileCheck %s

module {
  func.func public @test_reduce_prod_full_workaround(%arg0: tensor<128x10x32x4xf32>) -> tensor<f32> {
    // CHECK-LABEL: func.func public @test_reduce_prod_full_workaround
    %0 = ttir.empty() : tensor<f32>
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<128x10x32x4xf32
    // CHECK-SAME: -> tensor<128x10x32x4xbf16
    // CHECK: %[[PROD:[0-9]+]] = "ttnn.prod"(%[[ARG0]])
    // CHECK-SAME: {keep_dim = false}
    // CHECK-SAME: tensor<128x10x32x4xbf16,
    // CHECK-SAME: -> tensor<bf16,
    %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [0 : i32, 1 : i32, 2 : i32, 3 : i32], keep_dim = false}> : (tensor<128x10x32x4xf32>, tensor<f32>) -> tensor<f32>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[PROD]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<bf16
    // CHECK-SAME: -> tensor<f32
    return %1 : tensor<f32>
  }
}
