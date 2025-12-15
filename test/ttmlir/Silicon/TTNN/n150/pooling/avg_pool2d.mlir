// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  // Test 1: AvgPool2dOp without flattened compat info and with kernel specified
  func.func @avg_pool2d_non_flattened_with_kernel(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    // CHECK: ttnn.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }

  // Test 2: AvgPool2dOp without flattened compat info and with kernel and stride specified
  func.func @avg_pool2d_non_flattened_with_kernel_stride(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16> {
    // CHECK: ttnn.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16>
    return %1 : tensor<1x15x15x64xbf16>
  }

  // Test 3: AvgPool2dOp with flattened compat info and with kernel, stride and padding specified
  func.func @avg_pool2d_flattened_with_kernel_stride_padding(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    // CHECK: ttnn.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 4, 4>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 2, 2, 2, 2>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }

  // Test 4: AvgPool2dOp with flattened compat info and with kernel, stride, padding and ceil_mode specified
  func.func @avg_pool2d_flattened_with_kernel_stride_dilation_padding_ceil(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    // CHECK: ttnn.avg_pool2d
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 1, 1, 1, 1>,
      ceil_mode = true
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }

  // Test 5: AvgPool2dOp with count_include_pad = false
  func.func @avg_pool2d_count_include_pad_false(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    // CHECK: ttnn.avg_pool2d
    // CHECK: count_include_pad = false
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 4, 4>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 2, 2, 2, 2>,
      ceil_mode = false,
      count_include_pad = false
    }> : (tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }

  // Test 6: AvgPool2dOp with padding that exceeds half kernel size
  func.func @avg_pool2d_padding_exceeds_half_kernel_size(%arg0: tensor<1x1x1x1536xbf16>) -> tensor<1x1x64x1536xbf16> {
    // CHECK: ttnn.reshape
    // CHECK: ttnn.pad
    // CHECK: ttnn.reshape
    // CHECK: ttnn.avg_pool2d
    %0 = "ttir.avg_pool2d"(%arg0) <{
      ceil_mode = false,
      count_include_pad = true,
      dilation = array<i32: 1, 1>,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 1, input_width = 1>, kernel = array<i32: 8, 8>,
      padding = array<i32: 7, 7, 7, 7>,
      stride = array<i32: 1, 1>
    }> : (tensor<1x1x1x1536xbf16>) -> tensor<1x1x64x1536xbf16>
    return %0 : tensor<1x1x64x1536xbf16>
  }
}
