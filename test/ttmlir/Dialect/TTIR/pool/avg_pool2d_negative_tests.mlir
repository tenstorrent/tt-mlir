// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for avg_pool2d operation

// Test 1: AvgPool2dOp with invalid tensor rank (input)
module {
  func.func @avg_pool2d_invalid_input_rank(%arg0: tensor<1x1024xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op input must be a 4D tensor
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1024xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 2: AvgPool2dOp with invalid tensor rank (output)
module {
  func.func @avg_pool2d_invalid_output_rank(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x900xbf16> {
    %0 = ttir.empty() : tensor<1x900xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op output must be a 4D tensor
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      ceil_mode = false,
      padding = array<i32: 0, 0, 0, 0>,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x900xbf16>) -> tensor<1x900xbf16>
    return %1 : tensor<1x900xbf16>
  }
}

// -----
// Test 3: AvgPool2dOp with invalid FlattenedCompatInfoAttr
module {
  func.func @avg_pool2d_invalid_flattened_compat_info(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op the input tensor's flattened dimension (1024) does not match the product of batch_size * input_height * input_width from FlattenedCompatInfo (1 * 32 * 64 = 2048)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 64>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 4: AvgPool2dOp with invalid kernel
module {
  func.func @avg_pool2d_invalid_kernel(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op kernel size attribute values must be > 0
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: -1, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 5: AvgPool2dOp with invalid stride
module {
  func.func @avg_pool2d_invalid_stride(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op stride attribute values must be > 0
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: -1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 6: AvgPool2dOp with invalid dilation
module {
  func.func @avg_pool2d_invalid_dilation(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op dilation attribute values must be > 0
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: -1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 7: AvgPool2dOp with invalid padding
module {
  func.func @avg_pool2d_invalid_padding(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op padding attribute values must be >= 0
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: -1, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 8: AvgPool2dOp with effective kernel size larger than input size
module {
  func.func @avg_pool2d_effective_kernel_larger_than_input(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op effective kernel size (64, 64) cannot be greater than the padded input size per channel (32, 32)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 64, 64>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 10: AvgPool2dOp without flattened compat info and mismatch on batch size
module {
  func.func @avg_pool2d_no_flattened_compat_batch_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> tensor<2x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<2x30x30x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op batch size from the input tensor (1) must match the first dimension of the output tensor (2)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<2x30x30x64xbf16>) -> tensor<2x30x30x64xbf16>
    return %1 : tensor<2x30x30x64xbf16>
  }
}

// -----
// Test 11: AvgPool2dOp without flattened compat info and mismatch on channel size
module {
  func.func @avg_pool2d_no_flattened_compat_channel_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x128xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x128xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op number of output channels from the output tensor (128) must match the number of input channels (64)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x128xbf16>) -> tensor<1x30x30x128xbf16>
    return %1 : tensor<1x30x30x128xbf16>
  }
}

// -----
// Test 12: AvgPool2dOp without flattened compat info and mismatch on height dimension
module {
  func.func @avg_pool2d_no_flattened_compat_height_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x28x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x28x30x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op output tensor height and width dimension (28, 30) do not match the expected dimensions (30, 30)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x28x30x64xbf16>) -> tensor<1x28x30x64xbf16>
    return %1 : tensor<1x28x30x64xbf16>
  }
}

// -----
// Test 13: AvgPool2dOp without flattened compat info and mismatch on width dimension
module {
  func.func @avg_pool2d_no_flattened_compat_width_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x28x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x28x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op output tensor height and width dimension (30, 28) do not match the expected dimensions (30, 30)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x30x28x64xbf16>) -> tensor<1x30x28x64xbf16>
    return %1 : tensor<1x30x28x64xbf16>
  }
}

// -----
// Test 14: AvgPool2dOp without flattened compat info and mismatch on height and width dimensions
module {
  func.func @avg_pool2d_no_flattened_compat_height_width_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x28x28x64xbf16> {
    %0 = ttir.empty() : tensor<1x28x28x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op output tensor height and width dimension (28, 28) do not match the expected dimensions (30, 30)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x28x28x64xbf16>) -> tensor<1x28x28x64xbf16>
    return %1 : tensor<1x28x28x64xbf16>
  }
}

// -----
// Test 15: AvgPool2dOp with flattened compat info and mismatch on channel size
module {
  func.func @avg_pool2d_flattened_channel_mismatch(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x128xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x128xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op number of output channels from the output tensor (128) must match the number of input channels (64)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x128xbf16>) -> tensor<1x1x900x128xbf16>
    return %1 : tensor<1x1x900x128xbf16>
  }
}

// -----
// Test 16: AvgPool2dOp with flattened compat info and mismatch on flattened dimension
module {
  func.func @avg_pool2d_flattened_flatten_mismatch(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x1024x64xbf16>
    // CHECK: error: 'ttir.avg_pool2d' op output tensor's flattened dimension (1024) does not match the product of batch_size * output_height * output_width (1 * 30 * 30 = 900)
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xbf16>) -> tensor<1x1x1024x64xbf16>
    return %1 : tensor<1x1x1024x64xbf16>
  }
}

// -----
// Test 17: AvgPool2dOp with invalid kernel array size
module {
  func.func @avg_pool2d_invalid_kernel(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op Expected integer or pair of integers, got tuple of size 3 for kernel attribute
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 2, 2, 1>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 18: AvgPool2dOp with invalid stride array size
module {
  func.func @avg_pool2d_invalid_kernel(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op Expected integer or pair of integers, got tuple of size 3 for stride attribute
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 19: AvgPool2dOp with invalid dilation array size
module {
  func.func @avg_pool2d_invalid_kernel(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op Expected integer or pair of integers, got tuple of size 3 for dilation attribute
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// -----
// Test 20: AvgPool2dOp with invalid padding array size
module {
  func.func @avg_pool2d_invalid_kernel(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    // CHECK:  error: 'ttir.avg_pool2d' op Expected integer, pair, or tuple of size 4, but got tuple of size 5 for padding attribute
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0, 1>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}
