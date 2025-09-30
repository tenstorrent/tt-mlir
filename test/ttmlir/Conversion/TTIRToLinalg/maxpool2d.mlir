// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test 1: MaxPool2dOp without flattened compat info and with kernel specified
module {
  func.func @max_pool2d_non_flattened_with_kernel(%arg0: tensor<1x32x32x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    // CHECK: tosa.max_pool2d
    // CHECK-NOT: tensor.extract_slice
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
    // CHECK: tosa.max_pool2d
    // CHECK: tensor.extract_slice
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
