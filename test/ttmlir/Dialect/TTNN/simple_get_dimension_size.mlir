// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<13x21x3xf32>) -> tensor<1xi32> {
    %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x3xf32>) -> tensor<1xi32>
    // CHECK: = "ttnn.full"
    // CHECK-SAME: fill_value = 2.100000e+01 : f32
    return %0 : tensor<1xi32>
  }
}
