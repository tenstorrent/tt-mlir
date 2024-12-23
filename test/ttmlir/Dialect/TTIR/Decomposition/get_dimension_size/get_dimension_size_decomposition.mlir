// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s
module  {
  func.func @get_dimension_size_decomposition(%arg0: tensor<32x64x128xf32>) -> tensor<1xi32> {
    // CHECK: [[VAL:%.+]] = "ttir.constant"
    // CHECK-SAME: value = dense<128> : tensor<1xi32>
    // CHECK: return [[VAL]] : tensor<1xi32>
    %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 2 : i32}> : (tensor<32x64x128xf32>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
  }
}
