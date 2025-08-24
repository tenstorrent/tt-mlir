// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_clamp {
  func.func public @test_clamp_constant(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-LABEL: @test_clamp_constant
    %0 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %1 = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<4xf32>}> : () -> tensor<4xf32>
    %2 = ttir.empty() : tensor<4xf32>
    // CHECK: %{{[0-9]+}} = "ttir.clamp_scalar"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}>
    // CHECK-SAME: (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %3 = "ttir.clamp_tensor"(%arg0, %0, %1, %2) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %3 : tensor<4xf32>
  }

  func.func public @test_clamp_indirect_constant_reshape(%arg0: tensor<1x16xbf16>) -> tensor<1x16xbf16> {
    // CHECK-LABEL: @test_clamp_indirect_constant_reshape
    %0 = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "ttir.constant"() <{value = dense<6> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = ttir.empty() : tensor<1xbf16>
    %3 = "ttir.typecast"(%0, %2) : (tensor<1xf32>, tensor<1xbf16>) -> tensor<1xbf16>
    %4 = ttir.empty() : tensor<1xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32]}> : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %6 = ttir.empty() : tensor<1xbf16>
    %7 = "ttir.typecast"(%1, %6) : (tensor<1xi32>, tensor<1xbf16>) -> tensor<1xbf16>
    %8 = ttir.empty() : tensor<1xbf16>
    %9 = "ttir.reshape"(%7, %8) <{shape = [1 : i32]}> : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    %10 = ttir.empty() : tensor<1x16xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.clamp_scalar"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{max = 6.000000e+00 : f32, min = 3.000000e+00 : f32}>
    // CHECK-SAME: (tensor<1x16xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    %11 = "ttir.clamp_tensor"(%arg0, %5, %9, %10) : (tensor<1x16xbf16>, tensor<1xbf16>, tensor<1xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    return %11 : tensor<1x16xbf16>
  }

  func.func public @test_clamp_indirect_constant_broadcast(%arg0: tensor<1x32xbf16>) -> tensor<1x32xbf16> {
    // CHECK-LABEL: @test_clamp_indirect_constant_broadcast
    %0 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %1 = "ttir.constant"() <{value = dense<5.000000e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %2 = ttir.empty() : tensor<1x1xbf16>
    %3 = "ttir.reshape"(%0, %2) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %4 = ttir.empty() : tensor<1x32xbf16>
    %5 = "ttir.broadcast"(%3, %4) <{broadcast_dimensions = array<i64: 1, 32>}> : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %6 = ttir.empty() : tensor<1x1xbf16>
    %7 = "ttir.reshape"(%1, %6) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
    %8 = ttir.empty() : tensor<1x32xbf16>
    %9 = "ttir.broadcast"(%7, %8) <{broadcast_dimensions = array<i64: 1, 32>}> : (tensor<1x1xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %10 = ttir.empty() : tensor<1x32xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.clamp_scalar"(%arg0, %{{[0-9]+}})
    // CHECK-SAME: <{max = 5.000000e+00 : f32, min = 2.000000e+00 : f32}>
    // CHECK-SAME: (tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %11 = "ttir.clamp_tensor"(%arg0, %5, %9, %10) : (tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>
    return %11 : tensor<1x32xbf16>
  }

  func.func public @test_clamp_tensor(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-LABEL: @test_clamp_tensor
    %0 = ttir.empty() : tensor<4xf32>
    // CHECK: %{{[0-9]+}} = "ttir.clamp_tensor"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %1 = "ttir.clamp_tensor"(%arg0, %arg1, %arg2, %0) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  func.func public @test_clamp_tensor_constant(%arg0: tensor<1x16xbf16>, %arg1: tensor<1xbf16>) -> tensor<1x16xbf16> {
    // CHECK-LABEL: @test_clamp_tensor_constant
    %0 = "ttir.constant"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = ttir.empty() : tensor<1xbf16>
    %2 = "ttir.typecast"(%0, %1) : (tensor<1xf32>, tensor<1xbf16>) -> tensor<1xbf16>
    %3 = ttir.empty() : tensor<1xbf16>
    %4 = "ttir.reshape"(%2, %3) <{shape = [1 : i32]}> : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<1xbf16>
    // CHECK: [[MIN:[0-9]+]] = "ttir.broadcast"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: <{broadcast_dimensions = array<i64: 1, 16>}> :
    // CHECK-SAME: (tensor<1x1xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    // CHECK: [[MAX:[0-9]+]] = "ttir.broadcast"(%{{[0-9]+}}, %{{[0-9]+}})
    // CHECK-SAME: <{broadcast_dimensions = array<i64: 1, 16>}> :
    // CHECK-SAME: (tensor<1x1xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    %5 = ttir.empty() : tensor<1x16xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.clamp_tensor"(%arg0, %[[MIN]], %[[MAX]], %{{[0-9]+}})
    // CHECK-SAME: (tensor<1x16xbf16>, tensor<1x16xbf16>, tensor<1x16xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    %6 = "ttir.clamp_tensor"(%arg0, %4, %arg1, %5) : (tensor<1x16xbf16>, tensor<1xbf16>, tensor<1xbf16>, tensor<1x16xbf16>) -> tensor<1x16xbf16>
    return %6 : tensor<1x16xbf16>
  }
}
