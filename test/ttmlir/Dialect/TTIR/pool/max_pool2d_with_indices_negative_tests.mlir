// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for max_pool2d_with_indices operation

// Test 1: MaxPool2dWithIndicesOp with invalid tensor rank (input)
module {
  func.func @max_pool2d_with_indices_invalid_input_rank(%arg0: tensor<1x1024xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op input must be a 4D tensor
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1024xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 2: MaxPool2dWithIndicesOp with invalid tensor rank (output)
module {
  func.func @max_pool2d_with_indices_invalid_output_rank(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x900xbf16>, tensor<1x900xi32>) {
    %0 = ttir.empty() : tensor<1x900xbf16>
    %1 = ttir.empty() : tensor<1x900xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op output must be a 4D tensor
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      ceil_mode = false,
      padding = array<i32: 0, 0, 0, 0>,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x900xbf16>, tensor<1x900xi32>) -> (tensor<1x900xbf16>, tensor<1x900xi32>)
    return %2, %3 : tensor<1x900xbf16>, tensor<1x900xi32>
  }
}

// -----
// Test 3: MaxPool2dWithIndicesOp with invalid FlattenedCompatInfoAttr
module {
  func.func @max_pool2d_with_indices_invalid_flattened_compat_info(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x1860x64xbf16>, tensor<1x1x1860x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x1860x64xbf16>
    %1 = ttir.empty() : tensor<1x1x1860x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op the input tensor's flattened dimension (1024) does not match the product of batch_size * input_height * input_width from FlattenedCompatInfo (1 * 32 * 64 = 2048)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 64>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x1860x64xbf16>, tensor<1x1x1860x64xi32>) -> (tensor<1x1x1860x64xbf16>, tensor<1x1x1860x64xi32>)
    return %2, %3 : tensor<1x1x1860x64xbf16>, tensor<1x1x1860x64xi32>
  }
}

// -----
// Test 4: MaxPool2dWithIndicesOp with invalid kernel size
module {
  func.func @max_pool2d_with_indices_invalid_kernel_size(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op kernel size attribute values must be > 0
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: -1, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 5: MaxPool2dWithIndicesOp with invalid stride
module {
  func.func @max_pool2d_with_indices_invalid_stride(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op stride attribute values must be > 0
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: -1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 6: MaxPool2dWithIndicesOp with invalid dilation
module {
  func.func @max_pool2d_with_indices_invalid_dilation(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op dilation attribute values must be > 0
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: -1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 7: MaxPool2dWithIndicesOp with invalid padding
module {
  func.func @max_pool2d_with_indices_invalid_padding(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op padding attribute values must be >= 0
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: -1, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 8: MaxPool2dWithIndicesOp with effective kernel size larger than input size
module {
  func.func @max_pool2d_with_indices_effective_kernel_larger_than_input(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op effective kernel size (33, 33) cannot be greater than the padded input size per channel (32, 32)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 32, 32>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 9: MaxPool2dWithIndicesOp without flattened compat info and mismatch on batch size
module {
  func.func @max_pool2d_with_indices_no_flattened_compat_batch_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<2x30x30x64xbf16>, tensor<2x30x30x64xi32>) {
    %0 = ttir.empty() : tensor<2x30x30x64xbf16>
    %1 = ttir.empty() : tensor<2x30x30x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op batch size from the input tensor (1) must match the first dimension of the output tensor (2)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<2x30x30x64xbf16>, tensor<2x30x30x64xi32>) -> (tensor<2x30x30x64xbf16>, tensor<2x30x30x64xi32>)
    return %2, %3 : tensor<2x30x30x64xbf16>, tensor<2x30x30x64xi32>
  }
}

// -----
// Test 10: MaxPool2dWithIndicesOp without flattened compat info and mismatch on channel size
module {
  func.func @max_pool2d_with_indices_no_flattened_compat_channel_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<1x30x30x128xbf16>, tensor<1x30x30x128xi32>) {
    %0 = ttir.empty() : tensor<1x30x30x128xbf16>
    %1 = ttir.empty() : tensor<1x30x30x128xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op number of output channels from the output tensor (128) must match the number of input channels (64)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x128xbf16>, tensor<1x30x30x128xi32>) -> (tensor<1x30x30x128xbf16>, tensor<1x30x30x128xi32>)
    return %2, %3 : tensor<1x30x30x128xbf16>, tensor<1x30x30x128xi32>
  }
}

// -----
// Test 11: MaxPool2dWithIndicesOp without flattened compat info and mismatch on height dimension
module {
  func.func @max_pool2d_with_indices_no_flattened_compat_height_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<1x28x30x64xbf16>, tensor<1x28x30x64xi32>) {
    %0 = ttir.empty() : tensor<1x28x30x64xbf16>
    %1 = ttir.empty() : tensor<1x28x30x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op output tensor height and width dimension (28, 30) do not match the expected dimensions (30, 30)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x28x30x64xbf16>, tensor<1x28x30x64xi32>) -> (tensor<1x28x30x64xbf16>, tensor<1x28x30x64xi32>)
    return %2, %3 : tensor<1x28x30x64xbf16>, tensor<1x28x30x64xi32>
  }
}

// -----
// Test 12: MaxPool2dWithIndicesOp without flattened compat info and mismatch on width dimension
module {
  func.func @max_pool2d_with_indices_no_flattened_compat_width_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<1x30x28x64xbf16>, tensor<1x30x28x64xi32>) {
    %0 = ttir.empty() : tensor<1x30x28x64xbf16>
    %1 = ttir.empty() : tensor<1x30x28x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op output tensor height and width dimension (30, 28) do not match the expected dimensions (30, 30)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x30x28x64xbf16>, tensor<1x30x28x64xi32>) -> (tensor<1x30x28x64xbf16>, tensor<1x30x28x64xi32>)
    return %2, %3 : tensor<1x30x28x64xbf16>, tensor<1x30x28x64xi32>
  }
}

// -----
// Test 13: MaxPool2dWithIndicesOp without flattened compat info and mismatch on height and width dimensions
module {
  func.func @max_pool2d_with_indices_no_flattened_compat_height_width_mismatch(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<1x28x28x64xbf16>, tensor<1x28x28x64xi32>) {
    %0 = ttir.empty() : tensor<1x28x28x64xbf16>
    %1 = ttir.empty() : tensor<1x28x28x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op output tensor height and width dimension (28, 28) do not match the expected dimensions (30, 30)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x28x28x64xbf16>, tensor<1x28x28x64xi32>) -> (tensor<1x28x28x64xbf16>, tensor<1x28x28x64xi32>)
    return %2, %3 : tensor<1x28x28x64xbf16>, tensor<1x28x28x64xi32>
  }
}

// -----
// Test 14: MaxPool2dWithIndicesOp with flattened compat info and mismatch on channel size
module {
  func.func @max_pool2d_with_indices_flattened_channel_mismatch(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x128xbf16>, tensor<1x1x900x128xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x128xbf16>
    %1 = ttir.empty() : tensor<1x1x900x128xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op number of output channels from the output tensor (128) must match the number of input channels (64)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x128xbf16>, tensor<1x1x900x128xi32>) -> (tensor<1x1x900x128xbf16>, tensor<1x1x900x128xi32>)
    return %2, %3 : tensor<1x1x900x128xbf16>, tensor<1x1x900x128xi32>
  }
}

// -----
// Test 15: MaxPool2dWithIndicesOp with flattened compat info and mismatch on flattened dimension
module {
  func.func @max_pool2d_with_indices_flattened_flatten_mismatch(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x1024x64xbf16>
    %1 = ttir.empty() : tensor<1x1x1024x64xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op output tensor's flattened dimension (1024) does not match the product of batch_size * output_height * output_width (1 * 30 * 30 = 900)
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xi32>) -> (tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xi32>)
    return %2, %3 : tensor<1x1x1024x64xbf16>, tensor<1x1x1024x64xi32>
  }
}

// -----
// Test 16: MaxPool2dWithIndicesOp with invalid kernel array size
module {
  func.func @max_pool2d_with_indices_invalid_kernel(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op Expected integer or pair of integers, got tuple of size 3 for kernel attribute
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2, 1>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 17: MaxPool2dWithIndicesOp with invalid stride array size
module {
  func.func @max_pool2d_with_indices_invalid_stride(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op Expected integer or pair of integers, got tuple of size 3 for stride attribute
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 18: MaxPool2dWithIndicesOp with invalid dilation array size
module {
  func.func @max_pool2d_with_indices_invalid_dilation(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op Expected integer or pair of integers, got tuple of size 3 for dilation attribute
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 19: MaxPool2dWithIndicesOp with invalid padding array size
module {
  func.func @max_pool2d_with_indices_invalid_padding(%arg0: tensor<1x1x1024x64xbf16>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) {
    %0 = ttir.empty() : tensor<1x1x900x64xbf16>
    %1 = ttir.empty() : tensor<1x1x900x64xi32>
    // CHECK:  error: 'ttir.max_pool2d_with_indices' op Expected integer, pair, or tuple of size 4, but got tuple of size 5 for padding attribute
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0, 1>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>, tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>) -> (tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>)
    return %2, %3 : tensor<1x1x900x64xbf16>, tensor<1x1x900x64xi32>
  }
}

// -----
// Test 20: MaxPool2dWithIndicesOp with mismatched shapes between pooled values and indices
module {
  func.func @max_pool2d_with_indices_shape_mismatch(%arg0: tensor<1x4x4x2xbf16>) -> (tensor<1x2x2x2xbf16>, tensor<1x2x2x4xi32>) {
    %0 = ttir.empty() : tensor<1x2x2x2xbf16>
    %1 = ttir.empty() : tensor<1x2x2x4xi32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op Pooled values and indices must have the same shape
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x4x4x2xbf16>, tensor<1x2x2x2xbf16>, tensor<1x2x2x4xi32>) -> (tensor<1x2x2x2xbf16>, tensor<1x2x2x4xi32>)
    return %2, %3 : tensor<1x2x2x2xbf16>, tensor<1x2x2x4xi32>
  }
}

// -----
// Test 21: MaxPool2dWithIndicesOp with non-integer element type for indices  
module {
  func.func @max_pool2d_with_indices_non_integer_indices(%arg0: tensor<1x4x4x2xbf16>) -> (tensor<1x2x2x2xbf16>, tensor<1x2x2x2xf32>) {
    %0 = ttir.empty() : tensor<1x2x2x2xbf16>
    %1 = ttir.empty() : tensor<1x2x2x2xf32>
    // CHECK: error: 'ttir.max_pool2d_with_indices' op Indices result must have integer element type
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{
      kernel = array<i32: 2, 2>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x4x4x2xbf16>, tensor<1x2x2x2xbf16>, tensor<1x2x2x2xf32>) -> (tensor<1x2x2x2xbf16>, tensor<1x2x2x2xf32>)
    return %2, %3 : tensor<1x2x2x2xbf16>, tensor<1x2x2x2xf32>
  }
}