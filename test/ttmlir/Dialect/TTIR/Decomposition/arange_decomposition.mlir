// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[VAR_0:[0-9]+]] = "ttir.arange"() <{
    // CHECK-SAME: arange_dimension = 0 : i64, dtype = f32, end = 32 : si64, start = 0 : si64, step = 1 : si64
    // CHECK-SAME: }> : () -> tensor<32xf32>
    // CHECK: %[[VAR_1:[0-9]+]] = "ttir.broadcast"
    %1 = "ttir.arange"() <{start = 0: si64, end = 32: si64, step = 1: si64, arange_dimension = 1: i64, dtype = f32}> : () -> tensor<1x32x128x128xf32>
    return %1 : tensor<1x32x128x128xf32>
  }

  func.func @test_arange_multiply(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK: %[[ARANGE:[0-9]+]] = "ttir.arange"() <{
    // CHECK-SAME: arange_dimension = 0 : i64, dtype = f32, end = 32 : si64, start = 0 : si64, step = 1 : si64
    // CHECK-SAME: }> : () -> tensor<32xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[ARANGE]], %{{.*}}) <{
    // CHECK-SAME: shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]
    // CHECK-SAME: }> : (tensor<32xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
    // CHECK: %[[BROADCAST:[0-9]+]] = "ttir.broadcast"(%[[RESHAPE]], %{{.*}}) <{
    // CHECK-SAME: broadcast_dimensions = array<i64: 1, 1, 128, 128>
    // CHECK-SAME: }> : (tensor<1x32x1x1xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    %1 = "ttir.arange"() <{start = 0: si64, dtype = f32, end = 32: si64, step = 1: si64, arange_dimension = 1: i64}> : () -> tensor<1x32x128x128xf32>
    %dps = ttir.empty() : tensor<1x32x128x128xf32>
    %2 = "ttir.multiply"(%arg0, %1, %dps) : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %2 : tensor<1x32x128x128xf32>
  }
}
