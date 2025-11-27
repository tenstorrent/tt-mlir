// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<1x128x128x32xf32>) -> tensor<1x1x1x32xf32> {
    // CHECK: = "ttnn.global_avg_pool2d"
    %0 = "ttir.global_avg_pool2d"(%arg0) : (tensor<1x128x128x32xf32>) -> tensor<1x1x1x32xf32>
    return %0 : tensor<1x1x1x32xf32>
  }
}
