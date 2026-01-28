// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: GlobalAvgPool2dOp basic
module {
  func.func @global_avg_pool2d_basic(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x1x1x64xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    // CHECK-NOT: ttir.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0) : (tensor<1x32x32x64xbf16>) -> tensor<1x1x1x64xbf16>
    return %1 : tensor<1x1x1x64xbf16>
  }
}

// Test 2: GlobalAvgPool2dOp with different spatial dimensions
module {
  func.func @global_avg_pool2d_64x64(%arg0: tensor<1x64x64x128xbf16>) -> tensor<1x1x1x128xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    %1 = "ttir.global_avg_pool2d"(%arg0) : (tensor<1x64x64x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %1 : tensor<1x1x1x128xbf16>
  }
}

// Test 3: GlobalAvgPool2dOp with batch size > 1
module {
  func.func @global_avg_pool2d_batched(%arg0: tensor<4x16x16x32xbf16>) -> tensor<4x1x1x32xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    %1 = "ttir.global_avg_pool2d"(%arg0) : (tensor<4x16x16x32xbf16>) -> tensor<4x1x1x32xbf16>
    return %1 : tensor<4x1x1x32xbf16>
  }
}
