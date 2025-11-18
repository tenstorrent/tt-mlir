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

func.func @conv_transpose2d_simple(%arg0: tensor<1x64x64x256xbf16>, %arg1: tensor<256x256x16x16xbf16>, %arg2: tensor<1x1x1x256xbf16>) -> tensor<1x73x67x256xbf16> {
    %0 = ttir.empty() : tensor<1x73x67x256xbf16>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[CONV:[0-9]+]] = "ttir.conv_transpose2d"(%[[RESHAPE1]]
    // CHECK: #ttir<flattened_compat batch_size = 1, input_height = 64, input_width = 64>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[CONV]]
    %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0)
            <{
              stride = 1: i32,
              padding = array<i32: 3, 6, 3, 6>,
              output_padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32}
            > : (tensor<1x64x64x256xbf16>, tensor<256x256x16x16xbf16>, tensor<1x1x1x256xbf16>, tensor<1x73x67x256xbf16>) -> tensor<1x73x67x256xbf16>
    return %1 : tensor<1x73x67x256xbf16>
  }

module {
  func.func @max_pool2d_simple(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16>, %arg2: tensor<1x1x1x64xbf16>) -> tensor<1x30x30x64xbf16> {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[POOL:[0-9]+]] = "ttir.max_pool2d"(%[[RESHAPE1]]
    // CHECK: #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[POOL]]
    %1 = "ttir.max_pool2d"(%arg0, %0) <{kernel = array<i32: 3, 3>, stride = array<i32: 1, 1>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
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
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{kernel = array<i32: 3, 3>, stride = array<i32: 1, 1>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %1 : tensor<1x30x30x64xbf16>
  }
}

module {
  func.func @max_pool2d_with_indices_simple(%arg0: tensor<1x32x32x64xbf16>) -> (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>) {
    %0 = tensor.empty() : tensor<1x30x30x64xbf16>
    %1 = tensor.empty() : tensor<1x30x30x64xi32>
    // CHECK: %[[RESHAPE1:[0-9]+]] = "ttir.reshape"
    // CHECK: %[[RESULT:[a-zA-Z_0-9]+]], %[[RESULT_INDICES:[a-zA-Z_0-9]+]] = "ttir.max_pool2d_with_indices"(%[[RESHAPE1]]
    // CHECK: flattened_compat_info = #ttir<flattened_compat batch_size = 1, input_height = 32, input_width = 32>
    // CHECK: %[[RESHAPE2:[0-9]+]] = "ttir.reshape"(%[[RESULT]]
    // CHECK: %[[RESHAPE3:[0-9]+]] = "ttir.reshape"(%[[RESULT_INDICES]]
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1) <{kernel = array<i32: 3, 3>, stride = array<i32: 1, 1>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x32x32x64xbf16>, tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>) -> (tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>)
    return %2, %3 : tensor<1x30x30x64xbf16>, tensor<1x30x30x64xi32>
  }
}
