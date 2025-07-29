// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<1x128x128x32xf32>) -> tensor<1x64x64x32xf32> {
    %0 = ttir.empty() : tensor<1x64x64x32xf32>
    // CHECK: = "ttnn.avg_pool2d"
    %1 = "ttir.avg_pool2d"(%arg0, %0) <{kernel_height=2: si32, kernel_width=2: si32, stride_height=2: si32, stride_width=2: si32, dilation_height=1: si32, dilation_width=1: si32, ceil_mode=false, padding_left=0: si32, padding_right=0: si32, padding_top=0: si32, padding_bottom=0: si32}> : (tensor<1x128x128x32xf32>, tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32>
    return %1 : tensor<1x64x64x32xf32>
  }
}
