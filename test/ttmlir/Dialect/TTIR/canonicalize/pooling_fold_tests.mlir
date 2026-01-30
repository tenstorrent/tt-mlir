// RUN: ttmlir-opt -canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Test case: AvgPool2d with identity configuration should be folded.
  func.func @avg_pool2d_identity_fold(%arg0: tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32> {
    // CHECK-LABEL: @avg_pool2d_identity_fold
    // CHECK-NOT: ttir.avg_pool2d
    // CHECK: return %arg0
    %1 = "ttir.avg_pool2d"(%arg0) <{
        kernel = array<i32: 1, 1>,
        stride = array<i32: 1, 1>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>
    return %1 : tensor<1x32x32x16xf32>
  }

  // Test case: MaxPool2d with identity configuration should be folded.
  func.func @max_pool2d_identity_fold(%arg0: tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32> {
    // CHECK-LABEL: @max_pool2d_identity_fold
    // CHECK-NOT: ttir.max_pool2d
    // CHECK: return %arg0
    %1 = "ttir.max_pool2d"(%arg0) <{
        kernel = array<i32: 1, 1>,
        stride = array<i32: 1, 1>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32>
    return %1 : tensor<1x64x64x32xf32>
  }

  // Test case: AvgPool2d with non-identity configuration should NOT be folded.
  func.func @avg_pool2d_no_fold(%arg0: tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32> {
    // CHECK-LABEL: @avg_pool2d_no_fold
    // CHECK: ttir.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
        kernel = array<i32: 2, 2>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32>
    return %1 : tensor<1x16x16x16xf32>
  }

  // Test case: MaxPool2d with padding should NOT be folded.
  func.func @max_pool2d_with_padding_no_fold(%arg0: tensor<1x32x32x16xf32>) -> tensor<1x32x33x16xf32> {
    // CHECK-LABEL: @max_pool2d_with_padding_no_fold
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0) <{
        kernel = array<i32: 1, 1>,
        stride = array<i32: 1, 1>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 1, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x32x32x16xf32>) -> tensor<1x32x33x16xf32>
    return %1 : tensor<1x32x33x16xf32>
  }

  // Test case: AvgPool2d with stride != 1 should NOT be folded.
  func.func @avg_pool2d_stride_no_fold(%arg0: tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32> {
    // CHECK-LABEL: @avg_pool2d_stride_no_fold
    // CHECK: ttir.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
        kernel = array<i32: 1, 1>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32>
    return %1 : tensor<1x16x16x16xf32>
  }

  // Test case: MaxPool2d with dilation != 1 should NOT be folded.
  func.func @max_pool2d_dilation_no_fold(%arg0: tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32> {
    // CHECK-LABEL: @max_pool2d_dilation_no_fold
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0) <{
        kernel = array<i32: 1, 1>,
        stride = array<i32: 1, 1>,
        dilation = array<i32: 2, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false
    }> : (tensor<1x32x32x16xf32>) -> tensor<1x32x32x16xf32>
    return %1 : tensor<1x32x32x16xf32>
  }
}
