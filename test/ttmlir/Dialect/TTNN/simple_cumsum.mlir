// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module @moreh_cumsum attributes {} {
  func.func public @test_moreh_cumsum_dim0(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim0
    %0 = tensor.empty() : tensor<1x32x128x128xbf16>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 0 : i64
    // CHECK-SAME: tensor<1x32x128x128xbf16,
    // CHECK-SAME: tensor<1x32x128x128xbf16,
    // CHECK-SAME: -> tensor<1x32x128x128xbf16,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 0 : i64}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    return %1 : tensor<1x32x128x128xbf16>
  }

  func.func public @test_moreh_cumsum_dim1(%arg0: tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim1
    %0 = tensor.empty() : tensor<4x4x128x128xf32>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 1 : i64
    // CHECK-SAME: tensor<4x4x128x128xf32,
    // CHECK-SAME: tensor<4x4x128x128xf32,
    // CHECK-SAME: -> tensor<4x4x128x128xf32,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 1 : i64}> : (tensor<4x4x128x128xf32>, tensor<4x4x128x128xf32>) -> tensor<4x4x128x128xf32>
    return %1 : tensor<4x4x128x128xf32>
  }
}
