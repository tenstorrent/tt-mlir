// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_module_reshape attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @test_reshape(%arg0: tensor<1x64x64x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x4096x64xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK-LABEL: func.func public @test_reshape
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty
    // CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg0, %[[EMPTY]])
    // CHECK-SAME: shape = array<i64: 1, 1, 4096, 64>
    // CHECK-SAME: (tensor<1x64x64x64xf32>, tensor<1x1x4096x64xf32>) -> tensor<1x1x4096x64xf32>
    %0 = stablehlo.reshape %arg0 : (tensor<1x64x64x64xf32>) -> tensor<1x1x4096x64xf32>
    // CHECK: return %[[VAL]]
    return %0 : tensor<1x1x4096x64xf32>
  }

  func.func public @test_reshape_i64(%arg0: tensor<1x1xi64>) -> tensor<1xi64> {
    // CHECK-LABEL: func.func public @test_reshape_i64
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty
    // CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg0, %[[EMPTY]])
    // CHECK-SAME: shape = array<i64: 1>
    // CHECK-SAME: (tensor<1x1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %0 = stablehlo.reshape %arg0 : (tensor<1x1xi64>) -> tensor<1xi64>
    // CHECK: return %[[VAL]]
    return %0 : tensor<1xi64>
  }

  func.func public @test_reshape_i1(%arg0: tensor<2x7xi1>) -> tensor<7x2xi1> {
    // CHECK-LABEL: func.func public @test_reshape_i1
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty
    // CHECK: %[[VAL:[0-9]+]] = "ttir.reshape"(%arg0, %[[EMPTY]])
    // CHECK-SAME: shape = array<i64: 7, 2>
    // CHECK-SAME: (tensor<2x7xbf16>, tensor<7x2xbf16>) -> tensor<7x2xbf16>
    %0 = stablehlo.reshape %arg0 : (tensor<2x7xi1>) -> tensor<7x2xi1>
    // CHECK: return %[[VAL]]
    return %0 : tensor<7x2xi1>
  }
}
