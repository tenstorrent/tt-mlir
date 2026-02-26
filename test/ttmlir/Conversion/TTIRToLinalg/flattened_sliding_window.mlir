// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that the TTIRToLinalg conversion handles flattened sliding window ops
// directly, unflattening inputs and re-flattening outputs during conversion.

// Test 1: Conv2d with flattened input.
// The input (1,1,1024,64) is unflattened to (1,32,32,64), conv2d produces
// (1,30,30,64), which is flattened back to (1,1,900,64).
module {
  func.func @conv2d_flattened(%arg0: tensor<1x1x1024x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16> {
    // CHECK-LABEL: func.func @conv2d_flattened
    // CHECK: tosa.transpose
    // CHECK: tosa.reshape %arg0
    // CHECK-SAME: -> tensor<1x32x32x64xbf16>
    // CHECK: tosa.conv2d
    // CHECK-SAME: -> tensor<1x30x30x64xbf16>
    // CHECK: tosa.reshape
    // CHECK-SAME: -> tensor<1x1x900x64xbf16>
    // CHECK: return
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32,
              flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
            }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// Test 2: MaxPool2d with flattened input.
// The input (1,1,1024,64) is unflattened to (1,32,32,64), max_pool2d produces
// (1,30,30,64), which is flattened back to (1,1,900,64).
module {
  func.func @max_pool2d_flattened(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    // CHECK-LABEL: func.func @max_pool2d_flattened
    // CHECK: tosa.reshape %arg0
    // CHECK-SAME: -> tensor<1x32x32x64xbf16>
    // CHECK: linalg.pooling_nhwc_max
    // CHECK-SAME: -> tensor<1x30x30x64xbf16>
    // CHECK: tosa.reshape
    // CHECK-SAME: -> tensor<1x1x900x64xbf16>
    // CHECK: return
    %1 = "ttir.max_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}

// Test 3: AvgPool2d with flattened input.
// The input (1,1,1024,64) is unflattened to (1,32,32,64), avg_pool2d (via
// sum pooling + division) produces (1,30,30,64), flattened to (1,1,900,64).
module {
  func.func @avg_pool2d_flattened(%arg0: tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16> {
    // CHECK-LABEL: func.func @avg_pool2d_flattened
    // CHECK: tosa.reshape %arg0
    // CHECK-SAME: -> tensor<1x32x32x64xbf16>
    // CHECK: linalg.pooling_nhwc_sum
    // CHECK: linalg.div
    // CHECK: tosa.reshape
    // CHECK-SAME: -> tensor<1x1x900x64xbf16>
    // CHECK: return
    %1 = "ttir.avg_pool2d"(%arg0) <{
      kernel = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      padding = array<i32: 0, 0, 0, 0>,
      ceil_mode = false,
      count_include_pad = true,
      flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    }> : (tensor<1x1x1024x64xbf16>) -> tensor<1x1x900x64xbf16>
    return %1 : tensor<1x1x900x64xbf16>
  }
}
