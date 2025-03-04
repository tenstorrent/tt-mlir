// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @moreh_cumsum attributes {} {
  func.func public @test_moreh_cumsum_dim0(%arg0: tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim0
    %0 = tensor.empty() : tensor<1x32x128x128xf32>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 0 : i64
    // CHECK-SAME: tensor<1x32x128x128xf32,
    // CHECK-SAME: -> tensor<1x32x128x128xf32,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 0 : i64}> : (tensor<1x32x128x128xf32>, tensor<1x32x128x128xf32>) -> tensor<1x32x128x128xf32>
    return %1 : tensor<1x32x128x128xf32>
  }

  func.func public @test_moreh_cumsum_dim1(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim1
    %0 = tensor.empty() : tensor<4x4x128x128xf32>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 1 : i64
    // CHECK-SAME: tensor<4x4x128x128xf32,
    // CHECK-SAME: -> tensor<4x4x128x128xf32,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 1 : i64}> : (tensor<4x4x128x128xf32>, tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %1 : tensor<4x4x128x128xf32>
  }

  func.func public @test_moreh_cumsum_dim2(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim2
    %0 = tensor.empty() : tensor<4x4x128x128xf32>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 2 : i64
    // CHECK-SAME: tensor<4x4x128x128xf32,
    // CHECK-SAME: -> tensor<4x4x128x128xf32,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 2 : i64}> : (tensor<4x4x128x128xf32>, tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %1 : tensor<4x4x128x128xf32>
  }

  func.func public @test_moreh_cumsum_dim3(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim3
    %0 = tensor.empty() : tensor<4x4x128x128xf32>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 3 : i64
    // CHECK-SAME: tensor<4x4x128x128xf32,
    // CHECK-SAME: -> tensor<4x4x128x128xf32,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 3 : i64}> : (tensor<4x4x128x128xf32>, tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %1 : tensor<4x4x128x128xf32>
  }
}
