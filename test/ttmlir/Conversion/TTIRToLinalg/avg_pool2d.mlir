// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: AvgPool2dOp with 2x2 kernel and stride 1.
module {
  func.func @avg_pool2d_basic(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x31x31x64xbf16> {
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
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK: tensor.extract_slice
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
