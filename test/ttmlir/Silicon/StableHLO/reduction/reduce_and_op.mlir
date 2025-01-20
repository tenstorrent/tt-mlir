// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @jit_reduce_add attributes {} {
  func.func public @test_reduce_and_4to3dim(%arg0: tensor<128x10x32x4xi1>, %cst_0: tensor<i1>) -> tensor<128x10x32xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_4to3dim
    // CHECK: %[[CONST:[0-9+]]] = "ttnn.full"
    // CHECK-SAME: <{fillValue = 4.000000e+00 : f32}>
    // CHECK-SAME: -> [[TENSOR:tensor<128x10x32xbf16,]]
    // CHECK: %[[SUM:[0-9]+]] = "ttnn.sum"
    // CHECK-SAME: dim_arg = [3 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: -> tensor<128x10x32x1xbf16,
    // CHECK: %[[RES:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: %[[SUM]]
    // CHECK-SAME: <{shape = [128 : i32, 10 : i32, 32 : i32]}>
    // CHECK-SAME: tensor<128x10x32x1xbf16,
    // CHECK-SAME: -> [[TENSOR]]
    // CHECK: "ttnn.eq"
    // CHECK-SAME: %[[RES]], %[[CONST]]
    // CHECK-SAME: -> [[TENSOR]]
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [3] : (tensor<128x10x32x4xi1>, tensor<i1>) -> tensor<128x10x32xi1>
    return %0 : tensor<128x10x32xi1>
  }

  func.func public @test_reduce_and_3to2dim(%arg0: tensor<128x10x4xi1>, %cst_0: tensor<i1>) -> tensor<128x4xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_3to2dim
    // CHECK: %[[CONST:[0-9+]]] = "ttnn.full"
    // CHECK-SAME: <{fillValue = 1.000000e+01 : f32}>
    // CHECK-SAME: -> [[TENSOR:tensor<128x4xbf16,]]
    // CHECK: %[[SUM:[0-9]+]] = "ttnn.sum"
    // CHECK-SAME: dim_arg = [1 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: -> tensor<128x1x4xbf16,
    // CHECK: %[[RES:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: %[[SUM]]
    // CHECK-SAME: <{shape = [128 : i32, 4 : i32]}>
    // CHECK-SAME: tensor<128x1x4xbf16,
    // CHECK-SAME: -> [[TENSOR]]
    // CHECK: "ttnn.eq"
    // CHECK-SAME: %[[RES]], %[[CONST]]
    // CHECK-SAME: -> [[TENSOR]]
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [1] : (tensor<128x10x4xi1>, tensor<i1>) -> tensor<128x4xi1>
    return %0 : tensor<128x4xi1>
  }

  func.func public @test_reduce_and_2to1dim(%arg0: tensor<128x10xi1>, %cst_0: tensor<i1>) -> tensor<10xi1> {
    // CHECK-LABEL: func.func public @test_reduce_and_2to1dim
    // CHECK: %[[CONST:[0-9+]]] = "ttnn.full"
    // CHECK-SAME: <{fillValue = 1.280000e+02 : f32}>
    // CHECK-SAME: -> [[TENSOR:tensor<10xbf16,]]
    // CHECK: %[[SUM:[0-9]+]] = "ttnn.sum"
    // CHECK-SAME: dim_arg = [0 : i32]
    // CHECK-SAME: keep_dim = true
    // CHECK-SAME: -> tensor<1x10xbf16,
    // CHECK: %[[RES:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: %[[SUM]]
    // CHECK-SAME: <{shape = [10 : i32]}>
    // CHECK-SAME: tensor<1x10xbf16,
    // CHECK-SAME: -> [[TENSOR]]
    // CHECK: "ttnn.eq"
    // CHECK-SAME: %[[RES]], %[[CONST]]
    // CHECK-SAME: -> [[TENSOR]]
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.and across dimensions = [0] : (tensor<128x10xi1>, tensor<i1>) -> tensor<10xi1>
    return %0 : tensor<10xi1>
  }
}
