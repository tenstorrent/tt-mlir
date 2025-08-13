// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<13x21x3xf32>) -> tensor<1xi32> {
    %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x3xf32>) -> tensor<1xi32>
    // CHECK: [[VAL:%[0-9]+]] = "ttnn.full"(%{{[0-9]+}})
    // CHECK-SAME: fill_value = 21 : i32
    // CHECK-SAME: -> tensor<1xsi32
    return %0 : tensor<1xi32>
  }
}
