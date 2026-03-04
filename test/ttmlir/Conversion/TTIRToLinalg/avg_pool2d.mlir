// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: AvgPool2dOp with 2x2 kernel and stride 1.
module {
  func.func @avg_pool2d_basic(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x31x31x64xbf16> {
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: ttir.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      count_include_pad = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x31x31x64xbf16>
    return %1 : tensor<1x31x31x64xbf16>
  }
}

// Test 2: AvgPool2dOp with 3x3 kernel and stride 2.
module {
  func.func @avg_pool2d_stride2(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16> {
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      count_include_pad = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16>
    return %1 : tensor<1x15x15x64xbf16>
  }
}

// Test 3: AvgPool2dOp with padding and count_include_pad=true.
module {
  func.func @avg_pool2d_padding(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = false,
      count_include_pad = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    return %1 : tensor<1x32x32x64xbf16>
  }
}

// Test 4: AvgPool2dOp with padding and count_include_pad=false.
// Uses two pooling ops: one for sum, one for counting non-padded elements.
module {
  func.func @avg_pool2d_count_exclude_pad(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: ttir.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = false,
      count_include_pad = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    return %1 : tensor<1x32x32x64xbf16>
  }
}

// Test 5: AvgPool2dOp with non-1x1 dilation
module {
  // Input: 1x32x32x64, kernel: 3x3, dilation: 2x2, stride: 1x1, no padding
  // Dilated kernel size: (3-1)*2+1 = 5
  // Output: (32 - 5) / 1 + 1 = 28
  func.func @avg_pool2d_dilation(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x28x28x64xbf16> {
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: ttir.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      count_include_pad = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x28x28x64xbf16>
    return %1 : tensor<1x28x28x64xbf16>
  }
}

// Test 6: AvgPool2dOp with non-1x1 dilation and count_include_pad=false
module {
  // Input: 1x32x32x64, kernel: 3x3, dilation: 2x2, stride: 1x1, padding: 2x2x2x2
  // Dilated kernel size: (3-1)*2+1 = 5
  // Output: (32 + 2 + 2 - 5) / 1 + 1 = 32
  func.func @avg_pool2d_dilation_exclude_pad(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: ttir.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 2, 2, 2, 2>,
      ceil_mode = false,
      count_include_pad = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    return %1 : tensor<1x32x32x64xbf16>
  }
}

// Test 7: AvgPool2dOp with ceil_mode=true, count_include_pad=true
module {
  // Input: 1x32x32x64, kernel: 3x3, stride: 2x2, dilation: 1x1, no padding
  // ceil output: ceil(29/2) + 1 = 16
  // Extra padding is added to bottom and right to align dimensions.
  func.func @avg_pool2d_ceil_mode(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x16x16x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = true,
      count_include_pad = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x16x16x64xbf16>
    return %1 : tensor<1x16x16x64xbf16>
  }
}

// Test 8: AvgPool2dOp with ceil_mode=true, count_include_pad=false
module {
  // Input: 1x32x32x64, kernel: 3x3, stride: 2x2, dilation: 1x1, padding: 1x1x1x1
  // ceil output: ceil(31/2) + 1 = 17
  func.func @avg_pool2d_ceil_mode_exclude_pad(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = true,
      count_include_pad = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }
}
