// RUN: ttmlir-opt -ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func public @test_moreh_cumsum_dim2(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim2
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 1, 0, 3>
    // CHECK: "ttnn.moreh_cumsum"
    // CHECK-SAME: dim = 0
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 2, 1, 0, 3>
    %1 = "ttnn.moreh_cumsum"(%arg0) <{dim = 2 : i64}> : (tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %1 : tensor<4x4x128x128xf32>
  }

  func.func public @test_moreh_cumsum_dim3(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim3
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 3, 1, 2, 0>
    // CHECK: "ttnn.moreh_cumsum"
    // CHECK-SAME: dim = 0
    // CHECK: "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 3, 1, 2, 0>
    %1 = "ttnn.moreh_cumsum"(%arg0) <{dim = 3 : i64}> : (tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %1 : tensor<4x4x128x128xf32>
  }
}
