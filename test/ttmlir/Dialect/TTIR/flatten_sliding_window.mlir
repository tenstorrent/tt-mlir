// RUN: ttmlir-opt --ttir-flatten-sliding-window -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @conv2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[CONV:[0-9]+]] = "ttir.conv2d"(%[[RESHAPE1]]
    // CHECK: #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[CONV]]
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

module {
  func.func @max_pool2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[POOL:[0-9]+]] = "ttir.max_pool2d"(%[[RESHAPE1]]
    // CHECK: #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[POOL]]
    %1 = "ttir.max_pool2d"(%arg0, %0)
            <{
              stride_height = 1: si32,
              stride_width = 1: si32,
              padding_top = 0: si32,
              padding_bottom = 0: si32,
              padding_left = 0: si32,
              padding_right = 0: si32,
              dilation_height = 1: si32,
              dilation_width = 1: si32,
              kernel_height = 3: si32,
              kernel_width = 3: si32,
              ceil_mode = false
            }> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

module {
  func.func @avg_pool2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[POOL:[0-9]+]] = "ttir.avg_pool2d"(%[[RESHAPE1]]
    // CHECK: #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[POOL]]
    %1 = "ttir.avg_pool2d"(%arg0, %0)
            <{
              stride_height = 1: si32,
              stride_width = 1: si32,
              padding_top = 0: si32,
              padding_bottom = 0: si32,
              padding_left = 0: si32,
              padding_right = 0: si32,
              dilation_height = 1: si32,
              dilation_width = 1: si32,
              kernel_height = 3: si32,
              kernel_width = 3: si32,
              ceil_mode = false
            }> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}
