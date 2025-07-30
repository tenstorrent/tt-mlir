// RUN: ttmlir-opt --split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t
// Positive tests for max_pool2d operation

// Test 1: MaxPool2dOp without flattened compat info and with kernel specified
module {
  func.func @max_pool2d_non_flattened_with_kernel(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

// Test 2: MaxPool2dOp without flattened compat info and with kernel and stride specified
module {
  func.func @max_pool2d_non_flattened_with_kernel(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x15x15x64xbf16> {
    %0 = ttir.empty() : tensor<1x15x15x64xbf16>
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x15x15x64xbf16>) -> tensor<1x15x15x64xbf16>
    return %1 : tensor<1x15x15x64xbf16>
  }
}

// Test 3: MaxPool2dOp with flattened compat info and with kernel, stride and dilation specified
module {
  func.func @max_pool2d_flattened_with_kernel_stride_dilation(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x10x10x64xbf16> {
    %0 = ttir.empty() : tensor<1x10x10x64xbf16>
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 3, 3>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x10x10x64xbf16>) -> tensor<1x10x10x64xbf16>
    return %1 : tensor<1x10x10x64xbf16>
  }
}

// Test 4: MaxPool2dOp with flattened compat info and with kernel, stride, dilation and padding specified
module {
  func.func @max_pool2d_flattened_with_kernel_stride_dilation_padding(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x11x11x64xbf16> {
    %0 = ttir.empty() : tensor<1x11x11x64xbf16>
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 3, 3>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 1, 1, 2, 2>,
      ceil_mode = false
    }> : (tensor<1x32x32x64xbf16>, tensor<1x11x11x64xbf16>) -> tensor<1x11x11x64xbf16>
    return %1 : tensor<1x11x11x64xbf16>
  }
}

// Test 5: MaxPool2dOp with flattened compat info and with kernel, stride, dilation, padding and ceil_mode specified
module {
  func.func @max_pool2d_flattened_with_kernel_stride_dilation_padding(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x17x17x64xbf16> {
    %0 = ttir.empty() : tensor<1x17x17x64xbf16>
    // CHECK: ttir.max_pool2d
    %1 = "ttir.max_pool2d"(%arg0, %0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 2, 2>,
      dilation = array<i32: 2, 2>,
      padding = array<i32: 2, 2, 2, 2>,
      ceil_mode = true
    }> : (tensor<1x32x32x64xbf16>, tensor<1x17x17x64xbf16>) -> tensor<1x17x17x64xbf16>
    return %1 : tensor<1x17x17x64xbf16>
  }
}
