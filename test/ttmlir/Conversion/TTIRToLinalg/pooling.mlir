// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

//===----------------------------------------------------------------------===//
// MaxPool2d tests
//===----------------------------------------------------------------------===//

// MaxPool2dOp without flattened compat info and with kernel specified
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

// MaxPool2dOp without flattened compat info and with kernel and stride specified
module {
  func.func @max_pool2d_non_flattened_with_stride(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16> {
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

// MaxPool2dOp with non-1x1 dilation
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

// MaxPool2dOp with non-1x1 dilation and padding
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

// MaxPool2dOp with ceil_mode=true and stride > 1, no padding
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

// MaxPool2dOp with ceil_mode=true with padding
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

// MaxPool2dOp with f32 element type
module {
  func.func @max_pool2d_f32(%arg0: tensor<1x32x32x64xf32>) -> tensor<1x30x30x64xf32> {
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xf32>) -> tensor<1x30x30x64xf32>
    return %1 : tensor<1x30x30x64xf32>
  }
}

// MaxPool2dOp with i32 element type
module {
  func.func @max_pool2d_i32(%arg0: tensor<1x32x32x64xi32>) -> tensor<1x30x30x64xi32> {
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xi32>) -> tensor<1x30x30x64xi32>
    return %1 : tensor<1x30x30x64xi32>
  }
}

// MaxPool2dOp with i32 element type and padding
module {
  func.func @max_pool2d_i32_padding(%arg0: tensor<1x32x32x64xi32>) -> tensor<1x32x32x64xi32> {
    // CHECK: tensor.pad
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-NOT: tensor.extract_slice
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xi32>) -> tensor<1x32x32x64xi32>
    return %1 : tensor<1x32x32x64xi32>
  }
}

//===----------------------------------------------------------------------===//
// AvgPool2d tests
//===----------------------------------------------------------------------===//

// AvgPool2dOp with 2x2 kernel and stride 1.
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

// AvgPool2dOp with 3x3 kernel and stride 2.
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

// AvgPool2dOp with padding and count_include_pad=true.
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

// AvgPool2dOp with padding and count_include_pad=false.
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

// AvgPool2dOp with non-1x1 dilation
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

// AvgPool2dOp with non-1x1 dilation and count_include_pad=false
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

// AvgPool2dOp with ceil_mode=true, count_include_pad=true
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

// AvgPool2dOp with ceil_mode=true, count_include_pad=false
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

// AvgPool2dOp with f32 element type
module {
  func.func @avg_pool2d_f32(%arg0: tensor<1x32x32x64xf32>) -> tensor<1x31x31x64xf32> {
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
    }> : (tensor<1x32x32x64xf32>) -> tensor<1x31x31x64xf32>
    return %1 : tensor<1x31x31x64xf32>
  }
}

// AvgPool2dOp with i32 element type
module {
  func.func @avg_pool2d_i32(%arg0: tensor<1x32x32x64xi32>) -> tensor<1x31x31x64xi32> {
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
    }> : (tensor<1x32x32x64xi32>) -> tensor<1x31x31x64xi32>
    return %1 : tensor<1x31x31x64xi32>
  }
}

// AvgPool2dOp with i32 element type and padding, count_include_pad=false
module {
  func.func @avg_pool2d_i32_exclude_pad(%arg0: tensor<1x32x32x64xi32>) -> tensor<1x32x32x64xi32> {
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
    }> : (tensor<1x32x32x64xi32>) -> tensor<1x32x32x64xi32>
    return %1 : tensor<1x32x32x64xi32>
  }
}

//===----------------------------------------------------------------------===//
// GlobalAvgPool2d tests
//===----------------------------------------------------------------------===//

// GlobalAvgPool2dOp basic.
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

// GlobalAvgPool2dOp with different spatial dimensions.
module {
  func.func @global_avg_pool2d_64x64(%arg0: tensor<1x64x64x128xbf16>) -> tensor<1x1x1x128xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    %1 = "ttir.global_avg_pool2d"(%arg0) : (tensor<1x64x64x128xbf16>) -> tensor<1x1x1x128xbf16>
    return %1 : tensor<1x1x1x128xbf16>
  }
}

// GlobalAvgPool2dOp with batch size > 1.
module {
  func.func @global_avg_pool2d_batched(%arg0: tensor<4x16x16x32xbf16>) -> tensor<4x1x1x32xbf16> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    %1 = "ttir.global_avg_pool2d"(%arg0) : (tensor<4x16x16x32xbf16>) -> tensor<4x1x1x32xbf16>
    return %1 : tensor<4x1x1x32xbf16>
  }
}

// GlobalAvgPool2dOp with f32 element type.
module {
  func.func @global_avg_pool2d_f32(%arg0: tensor<1x32x32x64xf32>) -> tensor<1x1x1x64xf32> {
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.mul
    // CHECK-NOT: ttir.global_avg_pool2d
    %1 = "ttir.global_avg_pool2d"(%arg0) : (tensor<1x32x32x64xf32>) -> tensor<1x1x1x64xf32>
    return %1 : tensor<1x1x1x64xf32>
  }
}

//===----------------------------------------------------------------------===//
// MaxPool2dWithIndices tests
//===----------------------------------------------------------------------===//

// MaxPool2dWithIndicesOp basic - 2x2 kernel, stride 1, no padding.
module {
  // Input: 1x4x4x1, kernel: 2x2, stride: 1x1, no padding
  // Output: 1x3x3x1 (values and indices)
  func.func @max_pool2d_with_indices_basic(%arg0: tensor<1x4x4x1xbf16>) -> (tensor<1x3x3x1xbf16>, tensor<1x3x3x1xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_basic
    // CHECK: linalg.generic
    // CHECK-SAME: indexing_maps
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = tensor.empty() : tensor<1x3x3x1xbf16>
    %1 = tensor.empty() : tensor<1x3x3x1xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x4x4x1xbf16>, tensor<1x3x3x1xbf16>, tensor<1x3x3x1xi32>) -> (tensor<1x3x3x1xbf16>, tensor<1x3x3x1xi32>)
    return %2#0, %2#1 : tensor<1x3x3x1xbf16>, tensor<1x3x3x1xi32>
  }
}

// MaxPool2dWithIndicesOp with stride 2.
module {
  // Input: 1x8x8x64, kernel: 3x3, stride: 2x2, no padding
  // Output: (8 - 3) / 2 + 1 = 3 -> 1x3x3x64
  func.func @max_pool2d_with_indices_stride(%arg0: tensor<1x8x8x64xbf16>) -> (tensor<1x3x3x64xbf16>, tensor<1x3x3x64xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_stride
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x3x3x64xbf16>
    %1 = tensor.empty() : tensor<1x3x3x64xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x8x8x64xbf16>, tensor<1x3x3x64xbf16>, tensor<1x3x3x64xi32>) -> (tensor<1x3x3x64xbf16>, tensor<1x3x3x64xi32>)
    return %2#0, %2#1 : tensor<1x3x3x64xbf16>, tensor<1x3x3x64xi32>
  }
}

// MaxPool2dWithIndicesOp with padding.
module {
  // Input: 1x4x4x1, kernel: 3x3, stride: 1x1, padding: 1x1x1x1
  // Output: (4 + 1 + 1 - 3) / 1 + 1 = 4 -> 1x4x4x1
  func.func @max_pool2d_with_indices_padding(%arg0: tensor<1x4x4x1xbf16>) -> (tensor<1x4x4x1xbf16>, tensor<1x4x4x1xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_padding
    // CHECK: tensor.pad
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x4x4x1xbf16>
    %1 = tensor.empty() : tensor<1x4x4x1xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = false
    }> : (tensor<1x4x4x1xbf16>, tensor<1x4x4x1xbf16>, tensor<1x4x4x1xi32>) -> (tensor<1x4x4x1xbf16>, tensor<1x4x4x1xi32>)
    return %2#0, %2#1 : tensor<1x4x4x1xbf16>, tensor<1x4x4x1xi32>
  }
}

// MaxPool2dWithIndicesOp with ceil_mode=true.
module {
  // Input: 1x32x32x64, kernel: 3x3, stride: 2x2, no padding
  // ceil output: ceil(29/2) + 1 = 16
  func.func @max_pool2d_with_indices_ceil(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<1x16x16x64xbf16>, tensor<1x16x16x64xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_ceil
    // CHECK: tensor.pad
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x16x16x64xbf16>
    %1 = tensor.empty() : tensor<1x16x16x64xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = true
    }> : (tensor<1x32x32x64xbf16>, tensor<1x16x16x64xbf16>, tensor<1x16x16x64xi32>) -> (tensor<1x16x16x64xbf16>, tensor<1x16x16x64xi32>)
    return %2#0, %2#1 : tensor<1x16x16x64xbf16>, tensor<1x16x16x64xi32>
  }
}

// MaxPool2dWithIndicesOp with dilation.
module {
  // Input: 1x8x8x32, kernel: 3x3, dilation: 2x2, stride: 1x1, no padding
  // Dilated kernel size: (3-1)*2+1 = 5
  // Output: (8 - 5) / 1 + 1 = 4 -> 1x4x4x32
  func.func @max_pool2d_with_indices_dilation(%arg0: tensor<1x8x8x32xbf16>) -> (tensor<1x4x4x32xbf16>, tensor<1x4x4x32xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_dilation
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x4x4x32xbf16>
    %1 = tensor.empty() : tensor<1x4x4x32xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x8x8x32xbf16>, tensor<1x4x4x32xbf16>, tensor<1x4x4x32xi32>) -> (tensor<1x4x4x32xbf16>, tensor<1x4x4x32xi32>)
    return %2#0, %2#1 : tensor<1x4x4x32xbf16>, tensor<1x4x4x32xi32>
  }
}

// MaxPool2dWithIndicesOp with f32 element type
module {
  func.func @max_pool2d_with_indices_f32(%arg0: tensor<1x4x4x1xf32>) -> (tensor<1x3x3x1xf32>, tensor<1x3x3x1xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_f32
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x3x3x1xf32>
    %1 = tensor.empty() : tensor<1x3x3x1xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x4x4x1xf32>, tensor<1x3x3x1xf32>, tensor<1x3x3x1xi32>) -> (tensor<1x3x3x1xf32>, tensor<1x3x3x1xi32>)
    return %2#0, %2#1 : tensor<1x3x3x1xf32>, tensor<1x3x3x1xi32>
  }
}

// MaxPool2dWithIndicesOp with i32 element type
module {
  func.func @max_pool2d_with_indices_i32(%arg0: tensor<1x4x4x1xi32>) -> (tensor<1x3x3x1xi32>, tensor<1x3x3x1xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_i32
    // CHECK: linalg.generic
    // CHECK: arith.cmpi sgt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x3x3x1xi32>
    %1 = tensor.empty() : tensor<1x3x3x1xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x4x4x1xi32>, tensor<1x3x3x1xi32>, tensor<1x3x3x1xi32>) -> (tensor<1x3x3x1xi32>, tensor<1x3x3x1xi32>)
    return %2#0, %2#1 : tensor<1x3x3x1xi32>, tensor<1x3x3x1xi32>
  }
}

// MaxPool2dWithIndicesOp with i32 element type and padding
module {
  func.func @max_pool2d_with_indices_i32_padding(%arg0: tensor<1x4x4x1xi32>) -> (tensor<1x4x4x1xi32>, tensor<1x4x4x1xi32>) {
    // CHECK-LABEL: func.func @max_pool2d_with_indices_i32_padding
    // CHECK: tensor.pad
    // CHECK: linalg.generic
    // CHECK: arith.cmpi sgt
    // CHECK: arith.select
    %0 = tensor.empty() : tensor<1x4x4x1xi32>
    %1 = tensor.empty() : tensor<1x4x4x1xi32>
    %2:2 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = false
    }> : (tensor<1x4x4x1xi32>, tensor<1x4x4x1xi32>, tensor<1x4x4x1xi32>) -> (tensor<1x4x4x1xi32>, tensor<1x4x4x1xi32>)
    return %2#0, %2#1 : tensor<1x4x4x1xi32>, tensor<1x4x4x1xi32>
  }
}
