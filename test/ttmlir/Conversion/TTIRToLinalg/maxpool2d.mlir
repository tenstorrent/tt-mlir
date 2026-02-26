// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: MaxPool2dOp without flattened compat info and with kernel specified
module {
  func.func @max_pool2d_non_flattened_with_kernel(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// Test 2: MaxPool2dOp without flattened compat info and with kernel and stride specified
module {
  func.func @max_pool2d_non_flattened_with_kernel(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16> {
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16>
    return %1 : tensor<1x15x15x64xbf16>
  }
}

// Test 3: MaxPool2dOp with non-1x1 dilation
module {
  // Input: 1x32x32x64, kernel: 3x3, dilation: 2x2, stride: 1x1, no padding
  // Dilated kernel size: (3-1)*2+1 = 5
  // Output: (32 - 5) / 1 + 1 = 28
  func.func @max_pool2d_dilation(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x28x28x64xbf16> {
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x28x28x64xbf16>
    return %1 : tensor<1x28x28x64xbf16>
  }
}

// Test 4: MaxPool2dOp with non-1x1 dilation and padding
module {
  // Input: 1x32x32x64, kernel: 3x3, dilation: 2x2, stride: 1x1, padding: 2x2x2x2
  // Dilated kernel size: (3-1)*2+1 = 5
  // Output: (32 + 2 + 2 - 5) / 1 + 1 = 32
  func.func @max_pool2d_dilation_padding(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 2, 2, 2, 2>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x32x32x64xbf16>
    return %1 : tensor<1x32x32x64xbf16>
  }
}

// Test 5: MaxPool2dOp with ceil_mode=true and stride > 1, no padding
module {
  // Input: 1x32x32x64, kernel: 3x3, stride: 2x2, dilation: 1x1, no padding
  // ceil output: ceil(29/2) + 1 = 16
  // Extra padding is added to bottom and right to align dimensions.
  func.func @max_pool2d_ceil_mode(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x16x16x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x16x16x64xbf16>
    return %1 : tensor<1x16x16x64xbf16>
  }
}

// Test 6: MaxPool2dOp with ceil_mode=true with padding
module {
  // Input: 1x32x32x64, kernel: 3x3, stride: 2x2, dilation: 1x1, padding: 1x1x1x1
  // ceil output: ceil(31/2) + 1 = 17
  func.func @max_pool2d_ceil_mode_padding(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }
}
